import os
import time
import sqlite3
import argparse
import datetime

import numpy as np
from sqlite3 import Error

"""
Provides main and helper functions for processing and reformatting data
from the dense reconstruction.
"""

# General database handling routines for creating and populating tables
CAMERA_ROW_PARSER = [int, str, int, int, float, float, float, float]

IMAGE_ROW1_PARSER = [int, float, float, float, float, float, float, float, int, str]
IMAGE_ROW2_PARSER = [int, int, int]

POINT3D_ROW_PARSER = [int, float, float, float, int, int, int, float] 
POINT3D_TRACK_PARSER = [int, int]

def get_lines_from_textfile(txt_file_path):
    with open(txt_file_path) as txt_file:
        lines = txt_file.read().splitlines()
        txt_file.close()
        return lines

def create_reconstruction_database(db_file_path):
    conn = None
    try:
        conn = sqlite3.connect(str(db_file_path))
        print("sqlite version: {}".format(sqlite3.version))
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn

def create_all_tables(conn):
    sql_create_camera_table = """ CREATE TABLE IF NOT EXISTS cameras (
                                        camera_id INTEGER PRIMARY KEY,
                                        model TEXT NOT NULL,
                                        width INTEGER,
                                        heigth INTEGER,
                                        p1 FLOAT,
                                        p2 FLOAT,
                                        p3 FLOAT,
                                        p4 FLOAT
                                    ); """

    
    sql_create_point3d_table = """ CREATE TABLE IF NOT EXISTS points3d (
                                        point3d_id INTEGER PRIMARY KEY,
                                        x FLOAT,
                                        y FLOAT,
                                        z FLOAT,
                                        r INTEGER,
                                        g INTEGER,
                                        b INTEGER,
                                        error FLOAT
                                    ); """

    
    sql_create_image_table = """ CREATE TABLE IF NOT EXISTS images (                                        
                                        image_id INTEGER PRIMARY KEY,                                                                                
                                        qw FLOAT,
                                        qx FLOAT,
                                        qy FLOAT,
                                        qz FLOAT,
                                        x FLOAT,
                                        y FLOAT,
                                        z FLOAT,
                                        camera_id INTEGER,                                     
                                        name TEXT NOT NULL,
                                        FOREIGN KEY (camera_id) REFERENCES cameras (camera_id)
                                        ); """


    sql_create_points2d_table = """ CREATE TABLE IF NOT EXISTS points2d (
                                        point2d_id varchar(40) PRIMARY KEY,
                                        image_id INTEGER,
                                        point3d_id INTEGER,
                                        x FLOAT,
                                        y FLOAT,
                                        image_name TEXT NOT NULL,
                                        FOREIGN KEY (image_id) REFERENCES images (image_id), 
                                        FOREIGN KEY (point3d_id) REFERENCES points3d (point3d_id)                                    
                                    ); """

    CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
                                        keypoint_id TEXT PRIMARY KEY NOT NULL,
                                        image_name TEXT NOT NULL,
                                        x FLOAT,
                                        y FLOAT,
                                        data BLOB
                                    ); """


    cur = conn.cursor()
    cur.execute(sql_create_camera_table)
    cur.execute(sql_create_point3d_table)
    cur.execute(sql_create_image_table)
    cur.execute(sql_create_points2d_table)
    cur.execute(CREATE_KEYPOINTS_TABLE)

    conn.commit()
    cur.close()


# Helper populating functions (read text files and format information in database)
def populate_camera(conn, camera_txt_path):
    txt_lines = get_lines_from_textfile(camera_txt_path)        

    query = '''INSERT INTO cameras (camera_id, model, width, heigth, p1, p2, p3, p4) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'''
    values = None

    cur = conn.cursor()
    
    num_lines = len(txt_lines)
    for line in txt_lines[3:num_lines]:
        data_row_raw = line.split(' ')
        data_row_parsed = []
        # row indexes:     0        1       2      3       4
        # headers:     CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        for j in range(len(data_row_raw)):     
            data_row_parsed.append(CAMERA_ROW_PARSER[j](data_row_raw[j]))
        values = tuple(data_row_parsed)
            
        cur.execute(query, values)

    conn.commit()
    cur.close()

def populate_points3d(conn, points3d_txt_path):
    txt_lines = get_lines_from_textfile(points3d_txt_path)    

    query = '''INSERT INTO points3d (point3d_id, x, y, z, r, g, b, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'''
    values = None

    cur = conn.cursor()

    num_lines = len(txt_lines)
    for line in txt_lines[3:num_lines]:
        data_row_raw = line.split(' ')
        data_row_parsed = []
        for j in range(0, 8, 1):
            data_row_parsed.append(POINT3D_ROW_PARSER[j](data_row_raw[j]))
        values = tuple(data_row_parsed)
        cur.execute(query, values)
    
    conn.commit()
    cur.close()

def populate_image_and_points(conn, image_txt_path):
    txt_lines = get_lines_from_textfile(image_txt_path)    

    img_query = '''INSERT INTO images (image_id, qw, qx, qy, qz, x, y, z, camera_id, name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    point_query = '''INSERT INTO points2d (point2d_id, image_id, point3d_id, x, y, image_name) VALUES (?, ?, ?, ?, ?, ?)'''    

    cur = conn.cursor()
    num_lines = len(txt_lines)
    for line in range(4, num_lines, 2):
        data_row_raw = txt_lines[line].split(' ')
        data_points_2d = txt_lines[line + 1].split(' ')

        data_row_parsed = []
        
        for j in range(len(data_row_raw)):
            data_row_parsed.append(IMAGE_ROW1_PARSER[j](data_row_raw[j]))
        values = tuple(data_row_parsed)
        cur.execute(img_query, values)
        img_id = data_row_parsed[0]
        image_name = data_row_parsed[len(data_row_parsed)-1]

        for j in range(0, len(data_points_2d), 3):
            px = IMAGE_ROW2_PARSER[0](float(data_points_2d[j]))
            py = IMAGE_ROW2_PARSER[1](float(data_points_2d[j + 1]))
            point3d_id = IMAGE_ROW2_PARSER[2](data_points_2d[j + 2])
            
            if point3d_id == -1:
                continue

            point_id = str(datetime.datetime.now())
            values = (point_id, img_id, point3d_id, px, py, image_name)
            cur.execute(point_query, values)
            img_id = data_row_parsed[0]
    

    conn.commit()
    cur.close()

def fill_reco_db(db_path, camera_txt_path, points3d_txt_path, image_txt_path):
    start = time.process_time()
    conn = create_connection(db_path)
    populate_camera(conn=conn, camera_txt_path=camera_txt_path)
    populate_points3d(conn=conn, points3d_txt_path=points3d_txt_path)
    populate_image_and_points(conn, image_txt_path=image_txt_path)

    conn.close()
    print("Filled database in {} seconds".format(time.process_time() - start))

    return db_path

def create_db_tables(db_path):
    start = time.process_time()
    create_reconstruction_database(db_path)
    conn = create_connection(db_path)
    create_all_tables(conn)
    
    conn.close()
    print("SQLite INFO: create_full_db() in {} seconds ".format(time.process_time() - start))

    return db_path

# Database consulting class
class ReconstrunctionDB:
    def __init__(self, db_path) -> None:
        self.conn = create_connection(db_path)
        self.cur = self.conn.cursor()

    def _validate_empty_single_value(self, rows):
        if len(rows) == 0:
            return None
        return rows[0][0]

    def _execute_query(self, query, params):
        self.cur.execute(query, params)
        rows = self.cur.fetchall()
        return rows

    def array_to_blob(self, array):
        return array.tostring()
        
    def blob_to_array(self, blob, dtype, shape=(-1,)):
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

    def get_camera_params(self):
        query = '''SELECT p1, p2, p3, p4 from CAMERAS'''        
        rows = self._execute_query(query, ())
        return np.asarray(rows)

    def replace_dense_descriptor(self, desc_id, img_name, coor_x, coor_y, descriptor):
        try:
            query = '''DELETE FROM keypoints WHERE keypoint_id=?'''
            self.cur.execute(query, (desc_id,))
            self.conn.commit()
        except sqlite3.IntegrityError as err:
            pass
        
        try:
            data = self.array_to_blob(descriptor)
            query = '''INSERT INTO keypoints (keypoint_id, image_name, x, y, data) VALUES (?, ?, ?, ?, ?)'''
            values = (desc_id, img_name, coor_x, coor_y, data)
            self.cur.execute(query, values)
            self.conn.commit()     
        except sqlite3.IntegrityError as err:
            pass
            
    def get_image_names(self):
        query = '''SELECT name FROM images ORDER BY image_id ASC;'''
        rows = self._execute_query(query, ())
        return rows

    def get_keypoint_locations(self, image_name):
        query = '''SELECT kp.x, kp.y FROM keypoints kp WHERE kp.image_name=?'''        
        rows = self._execute_query(query, (image_name,))
        results = np.asarray(rows)

        return results.astype(float)

    def get_all_point3d_features(self):
        start = time.process_time()
        query = '''SELECT kp.image_name, kp.x, kp.y, kp.data, p3d.x, p3d.y, p3d.z FROM keypoints kp JOIN points2d p2d ON kp.image_name = p2d.image_name JOIN points3d p3d ON p2d.point3d_id = p3d.point3d_id WHERE kp.x =  p2d.x and kp.y =  p2d.y'''        
        rows = self._execute_query(query, ())
        image_names, image_coords, descriptor, coords_3d = [], [], [], []

        for row in rows:
            image_names.append(row[0])
            image_coords.append([row[1], row[2]])
            descriptor.append(self.blob_to_array(blob=row[3], dtype=np.float32))
            coords_3d.append([row[4], row[5], row[6]])

        print("SQLite INFO: get_all_point3d_features() retrieved {} points in {} seconds ".format(len(rows), time.process_time() - start))
        
                      #img names           # img coordinates                     # descriptor vector        # world 3d coordinates 
        return np.asarray(image_names), np.asarray(image_coords).astype(float), np.asarray(descriptor), np.asarray(coords_3d).astype(float) 

    def closeDB(self):
        self.cur.close()
        self.conn.close()

# Main functions for database population
def main(sequence_root, descriptor_type):
    if descriptor_type == "dense":
        print("Filling dreco_base_{}.db in {}...".format(args.matching_model_description, sequence_root))
        fill_reco_db(db_path=os.path.join(sequence_root, "dreco_base_{}.db".format(args.matching_model_description)), 
                     camera_txt_path=os.path.join(sequence_root, "colmap", "0", "cameras.txt"), 
                     points3d_txt_path=os.path.join(sequence_root, "colmap", "0", "points3D.txt"), 
                     image_txt_path=os.path.join(sequence_root, "colmap", "0", "images.txt"))
    else:
        print("Filling sift_dreco.db in {}...".format(sequence_root))
        fill_reco_db(db_path=os.path.join(sequence_root, "sift_dreco.db"), 
                     camera_txt_path=os.path.join(sequence_root, "initial_colmap", "0", "cameras.txt"), 
                     points3d_txt_path=os.path.join(sequence_root, "initial_colmap", "0", "points3D.txt"), 
                     image_txt_path=os.path.join(sequence_root, "initial_colmap", "0", "images.txt"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DRECO database management",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sequence_root", type=str, required=True,
                        help="root of the specific video sequence")
    parser.add_argument("--descriptor_type", type=str, default="dense", required=True,
                        help="type of descriptor used for database construction")
    parser.add_argument('--matching_model_description', type=str, 
                        help='description of model providing base for matching')

    args = parser.parse_args()

    main(sequence_root=args.sequence_root, descriptor_type=args.descriptor_type)