import sqlite3
import datetime
import numpy as np
from sqlite3 import Error


def create_localization_database(db_file_path):
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

def execute_query(db_path, query, params):
    conn = create_connection(db_path)
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()

    return rows

def create_table(conn):
    CREATE_CORRESPONDENCE_TABLE = """CREATE TABLE IF NOT EXISTS correspondences (
                                        image_name TEXT NOT NULL,
                                        point_id varchar(40) PRIMARY KEY,
                                        response FLOAT,
                                        x_2d FLOAT,
                                        y_2d FLOAT,
                                        x_3d FLOAT,
                                        y_3d FLOAT,
                                        z_3d FLOAT,
                                        preop_image_name TEXT NOT NULL,
                                        preop_x_2d FLOAT,
                                        preop_y_2d FLOAT
                                    ); """

    CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
                                        image_name TEXT NOT NULL,
                                        point_id varchar(40) PRIMARY KEY,
                                        x_2d_ref FLOAT,
                                        y_2d_ref FLOAT,
                                        x_3d_ref FLOAT,
                                        y_3d_ref FLOAT,
                                        z_3d_ref FLOAT
                                    ); """

    cur = conn.cursor()
    cur.execute(CREATE_CORRESPONDENCE_TABLE)
    cur.execute(CREATE_MATCHES_TABLE)

    conn.commit()
    cur.close()

def insert_image_correspondences(db_path, image_name, responses, points_2d, points_3d, preop_image_names, preop_image_coords):
    conn = create_connection(db_path)
    cur = conn.cursor()

    point_query = '''INSERT INTO correspondences (image_name, point_id, response, x_2d, y_2d, x_3d, y_3d, z_3d, preop_image_name, preop_x_2d, preop_y_2d) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''    

    for resp, p_2d, p_3d, preop_img, preop_coord in zip(responses, points_2d, points_3d, preop_image_names, preop_image_coords):
        point_id = str(datetime.datetime.now())
        values = (image_name, point_id, resp, p_2d[0].astype(float), p_2d[1].astype(float), p_3d[0], p_3d[1], p_3d[2], preop_img, preop_coord[0].astype(float), preop_coord[1].astype(float))
        cur.execute(point_query, values)

    conn.commit()
    cur.close()
    conn.close()

def insert_image_matches(db_path, image_name, points_2d_ref, points_3d_ref):
    conn = create_connection(db_path)
    cur = conn.cursor()

    point_query = '''INSERT INTO matches (image_name, point_id, x_2d_ref, y_2d_ref, x_3d_ref, y_3d_ref, z_3d_ref) VALUES (?, ?, ?, ?, ?, ?, ?)'''    

    for p_2d, p_3d in zip(points_2d_ref, points_3d_ref):
        point_id = str(datetime.datetime.now())
        values = (image_name, point_id, p_2d[0].astype(float), p_2d[1].astype(float), p_3d[0], p_3d[1], p_3d[2])
        cur.execute(point_query, values)

    conn.commit()
    cur.close()
    conn.close()

def retrieve_image_correspondences(db_path, query_image_name):
    query = "SELECT response, x_2d, y_2d, x_3d, y_3d, z_3d, preop_image_name, preop_x_2d, preop_y_2d FROM correspondences WHERE image_name=?"
    selection = execute_query(db_path=db_path, query=query, params=(query_image_name,))
    if len(selection) == 0:
        return None, None, None, None, None
    results = np.asarray(selection)
    
            # Responses   # Points 2D     # Points 3D     #Preop img   #Preop coords
    return results[:,0], results[:,1:3], results[:,3:6], results[:,6], results[:,7:9]

def retrieve_image_matches(db_path, query_image_name):
    query = "SELECT x_2d_ref, y_2d_ref, x_3d_ref, y_3d_ref, z_3d_ref FROM matches WHERE image_name=?"
    selection = execute_query(db_path=db_path, query=query, params=(query_image_name,))
    if len(selection) == 0:
        return None, None
    results = np.asarray(selection)
    
            # Points 2D     # Points 3D
    return results[:,0:2], results[:,3:6]

def retrieve_query_image_names(db_path):
    query = "SELECT DISTINCT image_name FROM correspondences"
    selection = execute_query(db_path=db_path, query=query, params=())
    
    return selection