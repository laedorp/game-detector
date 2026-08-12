# FORT-Cuh dataset attribution

This prepared dataset is derived from **FORT-Cuh v1**, exported from Roboflow
Universe on August 9, 2026 and provided by Roboflow user Aviles Joseph:

https://universe.roboflow.com/aviles-joseph/fort-cuh-mji4f

The source dataset identifies its license as **CC BY 4.0**:

https://creativecommons.org/licenses/by/4.0/

Preparation changes made by ProAim:

- remapped the source labels "0", "Fortnite", "Player"/"player", "bots",
  "enemy", "hello", "people", and "person" to one class named "player";
- excluded "head", "body", "ally", and "Yourself" annotations;
- excluded images with no retained full-player box;
- clipped retained bounding boxes to their image bounds;
- copied image bytes without visual modification; and
- retained the supplied split assignments while recording conservative
  cross-split original-file and `*_mp4-<frame>_jpg` video-sequence overlap.

The source archive hash and exact conversion counts are recorded in
manifest.json. Retain this notice when sharing the prepared data or a model
derived from it. The supplied validation/test splits are not independent of the
training sources, so their metrics may be optimistic. This notice describes the
dataset's stated license; it is not legal advice and does not replace review of
third-party model licenses.
