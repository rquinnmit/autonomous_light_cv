"""
Main pipeline loop.

Camera capture → motion detection → spatial translation → DMX output.
Runs on a dedicated thread, pushes display data to the UI via queue.
"""
