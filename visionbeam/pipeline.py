"""
Main pipeline loop.

Camera capture → person detection → multi-person tracking → motion
heatmap → spatial translation → DMX output. Runs on a dedicated thread,
pushes display data (frame, heatmap, tracked persons, aim state) to the
UI via queue.
"""
