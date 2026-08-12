from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from detection.hardware import (
    Accelerator,
    AcceleratorKind,
    HardwareProfile,
    ProcessorInfo,
    Vendor,
    describe,
    load_pci_names,
    parse_linux_cpuinfo,
    parse_windows_video_controllers,
    recommend,
    scan_hardware,
    scan_linux_accelerators,
)


CPUINFO = """processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Intel(R) Core(TM) i7-10850H CPU @ 2.70GHz
flags\t\t: fpu vme avx2 sse4_2 aes
"""


def write_pci_device(
    root: Path, slot: str, *, pci_class: str, vendor: str, device: str, vram: bool = False
) -> Path:
    entry = root / slot
    entry.mkdir(parents=True)
    (entry / "class").write_text(f"{pci_class}\n", encoding="utf-8")
    (entry / "vendor").write_text(f"{vendor}\n", encoding="utf-8")
    (entry / "device").write_text(f"{device}\n", encoding="utf-8")
    if vram:
        (entry / "mem_info_vram_total").write_text("17179869184\n", encoding="utf-8")
    return entry


class LinuxScanTests(unittest.TestCase):
    def test_cpuinfo_yields_model_name_and_feature_flags(self) -> None:
        processor = parse_linux_cpuinfo(CPUINFO, 12)

        self.assertIn("i7-10850H", processor.name)
        self.assertEqual(processor.logical_cores, 12)
        self.assertTrue(processor.has_wide_vectors)
        self.assertFalse(processor.has_int8_acceleration)

    def test_vnni_capable_cpu_is_reported_separately_from_avx2(self) -> None:
        processor = parse_linux_cpuinfo(
            CPUINFO.replace("avx2", "avx2 avx512_vnni"), 16
        )

        self.assertTrue(processor.has_int8_acceleration)

    def test_missing_cpuinfo_still_produces_a_usable_processor(self) -> None:
        processor = parse_linux_cpuinfo("", None)

        self.assertTrue(processor.name)
        self.assertEqual(processor.flags, frozenset())

    def test_display_and_accelerator_classes_are_recognized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pci_device(
                root, "0000:00:02.0", pci_class="0x030000", vendor="0x8086", device="0x9bc4"
            )
            write_pci_device(
                root, "0000:03:00.0", pci_class="0x030000", vendor="0x1002",
                device="0x73a5", vram=True,
            )
            write_pci_device(
                root, "0000:00:0b.0", pci_class="0x120000", vendor="0x8086", device="0x7d1d"
            )
            # A network card must never be mistaken for a compute device.
            write_pci_device(
                root, "0000:00:1f.6", pci_class="0x020000", vendor="0x8086", device="0x0d4f"
            )

            found = scan_linux_accelerators(root)

        kinds = [item.kind for item in found]
        self.assertEqual(kinds.count(AcceleratorKind.GPU), 2)
        self.assertEqual(kinds.count(AcceleratorKind.NPU), 1)
        self.assertEqual(len(found), 3)

    def test_dedicated_video_memory_marks_a_card_discrete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pci_device(
                root, "0000:03:00.0", pci_class="0x030000", vendor="0x1002",
                device="0x73a5", vram=True,
            )
            write_pci_device(
                root, "0000:00:02.0", pci_class="0x030000", vendor="0x8086", device="0x9bc4"
            )

            by_vendor = {item.vendor: item for item in scan_linux_accelerators(root)}

        self.assertIs(by_vendor[Vendor.AMD].discrete, True)
        self.assertIs(by_vendor[Vendor.INTEL].discrete, False)

    def test_unclassifiable_gpu_placement_stays_unknown_rather_than_guessing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_pci_device(
                root, "0000:01:00.0", pci_class="0x030000", vendor="0x10de", device="0x2482"
            )

            found = scan_linux_accelerators(root)

        self.assertIsNone(found[0].discrete)
        self.assertEqual(found[0].vendor, Vendor.NVIDIA)

    def test_missing_pci_root_is_not_an_error(self) -> None:
        self.assertEqual(scan_linux_accelerators(Path("/nonexistent-pci-root")), ())

    def test_pci_name_lookup_prefers_the_first_readable_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "pci.ids"
            database.write_text(
                "# comment\n8086  Intel Corporation\n\t9bc4  CometLake-H GT2\n"
                "1002  AMD\n\t73a5  Navi 21\n\t\t1002 0e3a  Subsystem\n",
                encoding="utf-8",
            )

            names = load_pci_names((root / "absent.ids", database))

        self.assertEqual(names[("8086", "9bc4")], "CometLake-H GT2")
        self.assertEqual(names[("1002", "73a5")], "Navi 21")
        # The doubly indented subsystem line must not become a device entry.
        self.assertEqual(len(names), 2)


class WindowsScanTests(unittest.TestCase):
    def test_video_controllers_are_parsed_with_vendor_and_placement(self) -> None:
        output = (
            "PCI\\VEN_1002&DEV_73A5&SUBSYS_0001|AMD Radeon RX 6950 XT|16777216000\n"
            "PCI\\VEN_8086&DEV_9BC4&SUBSYS_0002|Intel(R) UHD Graphics|1073741824\n"
        )

        found = parse_windows_video_controllers(output)

        self.assertEqual(len(found), 2)
        self.assertEqual(found[0].vendor, Vendor.AMD)
        self.assertIs(found[0].discrete, True)
        self.assertEqual(found[1].vendor, Vendor.INTEL)
        self.assertIsNone(found[1].discrete)

    def test_unparsable_adapter_memory_does_not_claim_a_placement(self) -> None:
        found = parse_windows_video_controllers(
            "PCI\\VEN_10DE&DEV_2482|NVIDIA GeForce RTX 3070|-1\n"
        )

        self.assertIsNone(found[0].discrete)
        self.assertEqual(found[0].vendor, Vendor.NVIDIA)

    def test_blank_and_malformed_lines_are_skipped(self) -> None:
        self.assertEqual(parse_windows_video_controllers("\n|\ngarbage\n"), ())

    def test_windows_scan_uses_the_injected_runner(self) -> None:
        calls: list[str] = []

        def runner(query: str) -> str:
            calls.append(query)
            return "PCI\\VEN_10DE&DEV_2482|NVIDIA GeForce RTX 3070|8589934592\n"

        profile = scan_hardware(
            system="Windows", logical_cores=8, powershell_runner=runner
        )

        self.assertEqual(len(calls), 1)
        gpus = profile.of_kind(AcceleratorKind.GPU)
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].vendor, Vendor.NVIDIA)


def build_profile(
    *,
    system: str = "linux",
    accelerators: tuple[Accelerator, ...] = (),
    runtime_devices: tuple[str, ...] = ("CPU",),
    flags: frozenset[str] = frozenset({"avx2"}),
) -> HardwareProfile:
    processor = ProcessorInfo(name="Intel(R) Core(TM) i7", logical_cores=12, flags=flags)
    cpu = Accelerator(kind=AcceleratorKind.CPU, vendor=Vendor.INTEL, name=processor.name)
    return HardwareProfile(
        system=system,
        processor=processor,
        accelerators=(cpu, *accelerators),
        runtime_devices=runtime_devices,
    )


class RecommendationTests(unittest.TestCase):
    def test_cpu_is_always_ready_and_int8_is_chosen_for_wide_vectors(self) -> None:
        plans = recommend(build_profile(), provider_factory=tuple)

        self.assertEqual(len(plans), 1)
        self.assertTrue(plans[0].ready)
        self.assertEqual(plans[0].backend, "openvino")
        self.assertEqual(plans[0].precision, "int8")

    def test_cpu_without_wide_vectors_keeps_fp32(self) -> None:
        plans = recommend(
            build_profile(flags=frozenset({"sse4_2"})), provider_factory=tuple
        )

        self.assertEqual(plans[0].precision, "fp32")

    def test_exposed_intel_gpu_outranks_the_cpu(self) -> None:
        gpu = Accelerator(
            kind=AcceleratorKind.GPU, vendor=Vendor.INTEL, name="UHD", discrete=False
        )
        plans = recommend(
            build_profile(accelerators=(gpu,), runtime_devices=("CPU", "GPU")),
            provider_factory=tuple,
        )

        self.assertEqual(plans[0].device, "GPU")
        self.assertTrue(plans[0].ready)
        self.assertEqual(plans[1].device, "CPU")

    def test_present_but_unexposed_intel_gpu_is_ranked_below_the_ready_cpu(self) -> None:
        gpu = Accelerator(
            kind=AcceleratorKind.GPU, vendor=Vendor.INTEL, name="UHD", discrete=False
        )
        plans = recommend(
            build_profile(accelerators=(gpu,), runtime_devices=("CPU",)),
            provider_factory=tuple,
        )

        self.assertEqual(plans[0].device, "CPU")
        self.assertFalse(plans[1].ready)
        self.assertIn("intel-compute-runtime", plans[1].setup_hint)

    def test_amd_gpu_never_recommends_openvino(self) -> None:
        gpu = Accelerator(
            kind=AcceleratorKind.GPU, vendor=Vendor.AMD, name="RX 6950 XT", discrete=True
        )
        plans = recommend(
            build_profile(accelerators=(gpu,)), provider_factory=tuple
        )

        amd = [plan for plan in plans if plan.accelerator.vendor is Vendor.AMD][0]
        self.assertEqual(amd.backend, "onnxruntime")
        self.assertNotEqual(amd.backend, "openvino")
        self.assertFalse(amd.ready)
        self.assertIn("ROCm", amd.setup_hint)

    def test_amd_gpu_on_windows_points_at_directml(self) -> None:
        gpu = Accelerator(
            kind=AcceleratorKind.GPU, vendor=Vendor.AMD, name="RX 6950 XT", discrete=True
        )
        plans = recommend(
            build_profile(system="windows", accelerators=(gpu,)), provider_factory=tuple
        )

        amd = [plan for plan in plans if plan.accelerator.vendor is Vendor.AMD][0]
        self.assertEqual(amd.device, "DmlExecutionProvider")
        self.assertIn("directml", amd.setup_hint.lower())

    def test_amd_gpu_becomes_ready_once_its_provider_is_installed(self) -> None:
        gpu = Accelerator(
            kind=AcceleratorKind.GPU, vendor=Vendor.AMD, name="RX 6950 XT", discrete=True
        )
        plans = recommend(
            build_profile(accelerators=(gpu,)),
            provider_factory=lambda: ("ROCMExecutionProvider", "CPUExecutionProvider"),
        )

        self.assertTrue(plans[0].ready)
        self.assertEqual(plans[0].accelerator.vendor, Vendor.AMD)
        self.assertEqual(plans[0].setup_hint, "")

    def test_nvidia_prefers_tensorrt_when_both_providers_exist(self) -> None:
        gpu = Accelerator(
            kind=AcceleratorKind.GPU, vendor=Vendor.NVIDIA, name="RTX 3070", discrete=True
        )
        plans = recommend(
            build_profile(accelerators=(gpu,)),
            provider_factory=lambda: (
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
            ),
        )

        self.assertEqual(plans[0].device, "TensorrtExecutionProvider")

    def test_intel_npu_is_ranked_first_when_openvino_exposes_it(self) -> None:
        npu = Accelerator(kind=AcceleratorKind.NPU, vendor=Vendor.INTEL, name="AI Boost")
        plans = recommend(
            build_profile(accelerators=(npu,), runtime_devices=("CPU", "NPU")),
            provider_factory=tuple,
        )

        self.assertEqual(plans[0].device, "NPU")
        self.assertTrue(plans[0].ready)

    def test_a_broken_provider_query_does_not_break_the_scan(self) -> None:
        gpu = Accelerator(
            kind=AcceleratorKind.GPU, vendor=Vendor.NVIDIA, name="RTX 3070", discrete=True
        )

        def explode() -> tuple[str, ...]:
            raise RuntimeError("onnxruntime is broken")

        plans = recommend(build_profile(accelerators=(gpu,)), provider_factory=explode)

        self.assertTrue(any(plan.ready for plan in plans))
        self.assertFalse(plans[-1].ready)

    def test_description_lists_every_device_and_its_hint(self) -> None:
        gpu = Accelerator(
            kind=AcceleratorKind.GPU, vendor=Vendor.AMD, name="RX 6950 XT", discrete=True
        )
        profile = build_profile(accelerators=(gpu,))

        text = describe(profile, recommend(profile, provider_factory=tuple))

        self.assertIn("RX 6950 XT (discrete)", text)
        self.assertIn("hint:", text)
        self.assertIn("Recommended order:", text)


if __name__ == "__main__":
    unittest.main()
