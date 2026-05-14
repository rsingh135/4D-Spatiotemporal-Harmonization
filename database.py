# This script is based on an original implementation by True Price.
# Created by liminghao; used by colmap.sh to inject known camera intrinsics.
import sys
import numpy as np
import sqlite3

IS_PYTHON3 = sys.version_info[0] >= 3


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params), camera_id),
        )
        return self.total_changes


def camTodatabase():
    import os
    import argparse

    camModelDict = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "SIMPLE_RADIAL": 2,
        "RADIAL": 3,
        "OPENCV": 4,
        "FULL_OPENCV": 5,
        "SIMPLE_RADIAL_FISHEYE": 6,
        "RADIAL_FISHEYE": 7,
        "OPENCV_FISHEYE": 8,
        "FOV": 9,
        "THIN_PRISM_FISHEYE": 10,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="database.db")
    parser.add_argument("--txt_path", type=str, default="colmap/sparse_cameras.txt")
    args = parser.parse_args()
    if not os.path.exists(args.database_path):
        print("ERROR: database path doesn't exist -- please check database.db.")
        return

    db = COLMAPDatabase.connect(args.database_path)

    idList = []
    modelList = []
    widthList = []
    heightList = []
    paramsList = []
    with open(args.txt_path, "r") as cam:
        lines = cam.readlines()
    for i in range(0, len(lines), 1):
        if lines[i][0] != "#":
            strLists = lines[i].split()
            cameraId = int(strLists[0])
            cameraModel = camModelDict[strLists[1]]
            width = int(strLists[2])
            height = int(strLists[3])
            paramstr = np.array(strLists[4:12])
            params = paramstr.astype(np.float64)
            # Drop unused slots for models with < 8 parameters
            if cameraModel == 0:  # SIMPLE_PINHOLE: f, cx, cy
                params = params[:3]
            idList.append(cameraId)
            modelList.append(cameraModel)
            widthList.append(width)
            heightList.append(height)
            paramsList.append(params)
            db.update_camera(cameraModel, width, height, params, cameraId)

    db.commit()
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0, len(idList), 1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])
    db.close()


if __name__ == "__main__":
    camTodatabase()
