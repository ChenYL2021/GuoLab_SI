
# pip install laspy scipy
#计算las投影xy的面积
import warnings

warnings.filterwarnings("ignore")
import os

os.environ['PROJ_LIB'] = r'D:/ProgramData/anaconda3/envs/envs/deepforest/Library/share/proj'  # 设置PROJ_LIB路径

import laspy
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from osgeo import osr
import pandas as pd


# 函数：计算LAS文件的凸包面积
def las_area(las_path):
    try:
        # 读取点云数据
        las_file = laspy.read(las_path)
        # 提取点的 xy 坐标
        points = np.vstack((las_file.x, las_file.y)).T
        # 仅保留二维坐标
        points_2d = points[:, :2]

        # 检查点的数量是否足够构成凸包（至少需要3个点）
        if len(points_2d) < 3:
            print(f"Not enough points to form a convex hull for {las_path}")
            return None

        # 计算凸包
        hull = ConvexHull(points_2d)
        # 使用凸包的顶点创建多边形
        hull_polygon = Polygon(points_2d[hull.vertices])
        # 计算凸包面积
        convex_hull_area = hull_polygon.area
        return convex_hull_area
    except Exception as e:
        print(f"Error processing {las_path}: {e}")
        return None


# 函数：获取LAS文件的中心坐标并转换为经纬度
def print_axis(las_file_name):
    try:
        # 打开las文件
        inFile = laspy.read(las_file_name)
        # 获取坐标范围
        x_max, y_max = inFile.header.max[0:2]
        x_min, y_min = inFile.header.min[0:2]
        x_now = (x_max + x_min) / 2
        y_now = (y_max + y_min) / 2

        # 定义UTM坐标系（zone 46）
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromProj4("+proj=utm +zone=46 +datum=WGS84 +units=m +no_defs")
        geosc1 = outRasterSRS.CloneGeogCS()
        cor_tran = osr.CoordinateTransformation(outRasterSRS, geosc1)
        coords = cor_tran.TransformPoint(x_now, y_now)

        return coords
    except Exception as e:
        print(f"Error processing {las_file_name}: {e}")
        return None


# 主程序
def main():
    all_las_directory = r"F:\DB\样本Lidar数据\zone_51"
    output_csv_file = r"G:\lidarDATA\zone46\las51_output.csv"

    data = []

    # 获取文件夹中的las文件路径列表
    las_files = [file for file in os.listdir(all_las_directory) if file.endswith('.las')]

    for las_file in las_files:
        las_path = os.path.join(all_las_directory, las_file)
        las_name = os.path.basename(las_path).split('.')[0]
        print(f"Processing {las_name}...")

        # 计算面积
        area = las_area(las_path)
        if area is not None:
            print("Area calculated")
        else:
            print(f"Skipping {las_name} due to processing error (area).")
            continue

        # 获取中心坐标

        coords = print_axis(las_path)
        if coords is not None:
            print("Coordinates calculated")
        else:
            print(f"Skipping {las_name} due to processing error (coordinates).")
            continue

        # 添加数据到列表
        data.append([coords[0], coords[1], las_name, area])

    # 创建 DataFrame 并保存到 CSV 文件
    Areadata = pd.DataFrame(data, columns=['Longitude', 'Latitude', 'Name', 'Area'])
    Areadata.to_csv(output_csv_file, index=False)
    print(f"转换后的数据已保存到 {output_csv_file}")


# 执行主程序
if __name__ == "__main__":
    main()
