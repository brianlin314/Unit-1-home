from scapy.all import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import os
from glob import glob

def zero_matrix(d): # 創建一個 d*d 的矩陣，並將其初始化為 0
    matrix = np.zeros((d,d), dtype=np.int_)
    return matrix

def create_submatrix(digit, w=16): # 將一個十六進位數字轉換為一個 2*2 的矩陣
    v = int(digit, base=16)*w+w/2
    return [[v, v],
            [v, v]]

def time_str_converter(pkttime): # 將封包的時間戳記轉換為人類可讀的格式 
    pkttime = int(pkttime)
    Y = str(datetime.fromtimestamp(pkttime).year)
    M = str(datetime.fromtimestamp(pkttime).strftime('%m'))
    D = str(datetime.fromtimestamp(pkttime).strftime('%d'))
    h = str(datetime.fromtimestamp(pkttime).strftime('%H'))
    m = str(datetime.fromtimestamp(pkttime).strftime('%M'))
    s = str(datetime.fromtimestamp(pkttime).strftime('%S'))
    n = str(pkttime)[11:]
    time = Y+M+D+h+m+s+n
    print(time)
    return time

def count_packets(pcap_path): # 計算 pcap 中的封包數量
    scapy_cap = rdpcap(pcap_path)
    print(f"Total packets in {pcap_path} : {len(scapy_cap)}")
    return len(scapy_cap)

def print_packet_info(pcap_path): # 打印 pcap 中的每個封包的信息
    packets = rdpcap(pcap_path)
    for packet in packets:
        raw_bytes = bytes(packet)
        print(len(raw_bytes))

def ip_masking(packet): # 隱藏封包中的源IP和目的IP
    if "IP" in packet:
        packet["IP"].src = "0.0.0.0"
        packet["IP"].dst = "0.0.0.0"
    return packet

def process_pcap_files(source_folder, output_folder):
    # 确保输出目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    # 遍历 source_folder 下的所有文件和目录
    for root, dirs, files in os.walk(source_folder):
        # output_path = output_folder + files + "/"
        # print("root:", root)
        for file in files:
            if file.endswith(".pcap"):
                # 使用 os.path.splitext() 函数去掉扩展名
                attack_num, _ = os.path.splitext(file)
                dir_name = os.path.join(output_folder, os.path.basename(os.path.normpath(root)), attack_num)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                pcap_path = os.path.join(root, file)
                # 对每个找到的 .pcap 文件调用 packet_encoder 函数
                packet_encoder(pcap_path, dir_name)

def copy_subdirectories_only_if_not_exist(src, dst):
    """
    只有当目标目录中不存在同名目录时，才复制源目录src下的所有子目录到目标目录dst。
    """
    # 确保目标目录存在
    os.makedirs(dst, exist_ok=True)
    
    # 遍历源目录
    for item in os.listdir(src):
        # 构建完整的源路径
        src_path = os.path.join(src, item)
        # 构建完整的目标路径
        dst_path = os.path.join(dst, item)
        
        # 如果是目录并且在目标目录中不存在同名目录，则创建目录
        if os.path.isdir(src_path) and not os.path.exists(dst_path):
            os.makedirs(dst_path)

#-------------------------------------------------------#
#                                                       #
#   INPUT : PCAP檔路徑，其中包含n個packets                 #
#   OUTPUT : PCAP中，每個封包encoding過後的matrix list     #
#   OUTPUT_SHAPE: (n, 3, 64, 64)                        #
#                                                       #
#-------------------------------------------------------#
def packet_encoder(pcap_path, img_output_path):
    np.set_printoptions(threshold=np.inf) # 設置 numpy 的打印選項，以便打印所有的數據
    count_packets(pcap_path) # 計算 pcap 中的封包數量，並打印出來
    scapy_cap = rdpcap(pcap_path) 
    for p, pkt in enumerate(scapy_cap): # 使用 enumerate 函數遍歷 scapy_cap 中的每個封包，p 是封包的索引，pkt 是封包對象。
        ip_masking(pkt) # 隱藏封包中的源IP和目的IP
        # time_str = time_str_converter(pkt.time) # 將封包的時間戳記轉換為人類可讀的格式
        byte_value = bytes(pkt) # 將封包轉換為 bytes 類型
        hex_value = "".join([hex(x)[2:].zfill(2) for x in byte_value]) # 將 bytes 類型的封包轉換為 16 進位的字符串
        matrix = zero_matrix(128) # 創建一個 128*128 的矩陣，並將其初始化為 0
            
        for i, digit in enumerate(hex_value):
            submatrix = create_submatrix(digit)
            j = i // 64
            k = i % 64
            try:
                matrix[j*2 : j*2+2, k*2 : k*2+2] = submatrix
            except:
                print("超過範圍限制!")
                break

        # 保存矩陣為圖片
        print(f"Saving {img_output_path}...")
        pkt_fig_name = img_output_path + '/' + str(p) + '.png'
        plt.imsave(pkt_fig_name, matrix, cmap='gray', vmin=0, vmax=255) # 將矩陣保存為圖片

if __name__ == "__main__":
    source_folder = '/SSD/p76111262/splited_pcap/'
    output_folder = '/SSD/p76111262/CIC-IDS-2018-v1/'
    
    copy_subdirectories_only_if_not_exist(source_folder, output_folder)
    process_pcap_files(source_folder, output_folder)   