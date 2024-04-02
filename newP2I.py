from scapy.all import rdpcap, IP, TCP, UDP, ARP, DNS, ICMP
from scapy.layers.tls.all import TLS, TLSClientHello
from scapy.layers.isakmp import ISAKMP
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 定义颜色映射
protocol_colors = {
    "HTTP": (0, 255, 0), # 绿色，代表Web流量的常见选择
    "TCP": (128, 128, 128), # 灰色，因为TCP是许多协议的基础，用中性色表示
    "UDP": (255, 255, 0), # 黄色，与TCP形成对比，表示另一种基础传输协议
    "SSH": (0, 191, 255), # 深天蓝，表示安全的远程访问
    "FTP": (255, 69, 0), # 红橙色，用于文件传输
    "DNS": (0, 0, 255), # 蓝色，常用于表示网络服务
    "ARP": (255, 165, 0), # 橙色，因为它与硬件地址相关
    "ICMP": (255, 0, 0), # 红色，表示控制消息和错误消息
    "TLS": (0, 255, 255), # 青色，代表安全的通信
    "ISAKMP": (192, 192, 192), # 银色，表示安全协议
    "SMTP": (255, 255, 0), # 黄色，用于电子邮件传输
    "POP3": (255, 192, 203), # 粉红色，邮件获取协议
    "IMAP": (173, 216, 230), # 浅蓝色，另一邮件获取协议
    "RDP": (64, 224, 208), # 绿松石，远程桌面协议
    "SMB": (128, 0, 0), # 暗红色，文件共享和打印服务
    "LLMNR": (255, 105, 180), # 热粉色，本地链接多播名称解析
    "RDPUDP": (128, 0, 128), # 紫色，RDP的UDP版本
    "WHOIS": (218, 112, 214), # 兰花紫，域名查询服务
    "NBMS/NBNS": (210, 105, 30), # 巧克力色，NetBIOS服务
    "mDNS": (244, 164, 96), # 沙棕色，多播DNS
    "SSDP": (46, 139, 87), # 海绿色，简单服务发现协议
    "Other": (0, 0, 0) # 黑色，表示其他协议
}

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

def create_submatrix(value, w=16):
    # 转换一个值为灰度值，返回RGB三通道相同的值
    v = value * w + w / 2
    return [v, v, v]

def create_color_from_byte(value):
    """
    根据字节值生成颜色。
    这个示例将直接使用字节值来影响颜色的生成。
    参数:
    - value: 字节值，范围在0到255之间。
    返回:
    - 一个包含RGB颜色的列表。
    """
    # 直接使用字节值作为R通道，通过某种计算影响G和B通道
    R = value
    G = 255 - value  # G通道使用反向值，以产生不同的颜色效果
    B = (value + 128) % 256  # B通道通过偏移和模运算产生变化

    return [R, G, B]

def identify_protocol(packet):
    # 根据封包内容确定协议类型
    # 此处只是示例，根据需要添加更多协议的判断逻辑
    protocol = "Other"
    if packet.haslayer(DNS):
        protocol = "DNS"
    elif packet.haslayer(ARP):
        protocol = "ARP"
    elif packet.haslayer(ICMP):
        protocol = "ICMP"
    elif packet.haslayer(TLS) or packet.haslayer(TLSClientHello):
        protocol = "TLS"
    elif packet.haslayer(ISAKMP):
        protocol = "ISAKMP"
    elif packet.haslayer(TCP):
        protocol = "TCP"
        if packet.dport == 80 or packet.sport == 80:
            protocol = "HTTP"
        elif packet.dport == 443 or packet.sport == 443:
            protocol = "HTTPS"
        elif packet.dport == 21 or packet.sport == 21:
            protocol = "FTP"
        elif packet.dport == 22 or packet.sport == 22:
            protocol = "SSH"
        elif packet.dport == 25 or packet.sport == 25:
            protocol = "SMTP"
        elif packet.dport == 110 or packet.sport == 110:
            protocol = "POP3"
        elif packet.dport == 143 or packet.sport == 143:
            protocol = "IMAP"
        elif packet.dport == 3389 or packet.sport == 3389:
            protocol = "RDP"
        elif packet.dport == 445 or packet.sport == 445 or packet.dport == 139 or packet.sport == 139:
            protocol = "SMB"
        elif packet.dport == 43 or packet.sport == 43:
            protocol = "WHOIS"
        elif packet.dport == 139 or packet.sport == 139:
            protocol = "NBMS"
    elif packet.haslayer(UDP):
        protocol = "UDP"
        if packet.dport == 53 or packet.sport == 53:
            protocol = "DNS"
        elif packet.dport == 67 or packet.dport == 68 or packet.sport == 67 or packet.sport == 68:
            protocol = "DHCP"
        elif packet.dport == 5355 or packet.sport == 5355:
            protocol = "LLMNR"
        elif packet.dport == 3389 or packet.sport == 3389:
            protocol = "RDPUDP"
        elif packet.dport == 137 or packet.sport == 137:
            protocol = "NBNS"
        elif packet.dport == 5353 or packet.sport == 5353:
            protocol = "mDNS"
        elif packet.dport == 1900 or packet.sport == 1900:
            protocol = "SSDP"
    return protocol

def normalize_time_diff(time_diff_ms, max_diff=60000):  # 假设最大时间差为1分钟（60000毫秒）
    # 归一化到[0, 255]
    if time_diff_ms > max_diff:
        time_diff_ms = max_diff
    normalized_value = int((time_diff_ms / max_diff) * 255)
    return normalized_value

# def packet_encoder(pcap_path, img_output_path):
#     packets = rdpcap(pcap_path)
#     base_time = packets[0].time  # 获取第0个封包的时间戳作为基准

#     for i, packet in enumerate(packets):
#         protocol = identify_protocol(packet)
#         color = protocol_colors.get(protocol, (0, 0, 0))  # 根据协议获取颜色
#         matrix = np.zeros((128, 128, 3), dtype=np.uint8)  # 创建图像矩阵

#         matrix[:6, :, :] = color  # 填充前6行为协议颜色

#         # 计算与第0个封包的时间差，并归一化
#         time_diff_ms = int((packet.time - base_time) * 1000)
#         normalized_time_diff = normalize_time_diff(time_diff_ms)
#         # print(f"Packet {i} time diff: {time_diff_ms}ms, normalized: {normalized_time_diff}")

#         # 在第7、8行填充归一化后的时间差灰度值
#         matrix[6:7, :, :] = [normalized_time_diff] * 3  # 填充灰度值

#         byte_value = bytes(packet)
#         hex_value = "".join([hex(x)[2:].zfill(2) for x in byte_value])
#         for j, digit in enumerate(hex_value[:8064]):
#             submatrix = create_submatrix(int(digit, 16))
#             row = j // 128 + 9  # 从第10行开始填充
#             col = j % 128
#             if row >= 128:  # 确保不会超出图像范围
#                 break
#             color = create_color_from_byte(byte)  # 为每个字节生成颜色
#             matrix[row, col, :] = color

#         # 保存图像
#         img_path = os.path.join(img_output_path, f"packet_{i}.png")
#         plt.imsave(img_path, matrix)
def packet_encoder(pcap_path, img_output_path):
    packets = rdpcap(pcap_path)
    base_time = packets[0].time  # 获取第0个封包的时间戳作为基准

    for i, packet in enumerate(packets):
        matrix = np.zeros((128, 128, 3), dtype=np.uint8)  # 创建图像矩阵

        # 设置协议颜色（前两行）
        protocol = identify_protocol(packet)
        protocol_color = protocol_colors.get(protocol, (0, 0, 0))  # 根据协议获取颜色
        matrix[:3, :, :] = protocol_color  # 填充前两行为协议颜色

        # 填充归一化后的时间差灰度值到第4、5行
        time_diff_ms = int((packet.time - base_time) * 1000)
        normalized_time_diff = normalize_time_diff(time_diff_ms)
        matrix[3:6, :, :] = [normalized_time_diff] * 3  # 填充灰度值

        # 根据字节值以彩色填充剩余的图像（从第6行开始）
        byte_value = bytes(packet)
        for j, byte in enumerate(byte_value):
            row = j // 128 + 9  # 从第6行开始填充
            if row >= 128:  # 确保不会超出图像范围
                break
            col = j % 128
            matrix[row, col, :] = create_color_from_byte(byte)  # 为每个字节生成颜色并填充

        # 保存图像
        img_path = os.path.join(img_output_path, f"packet_{i}.png")
        plt.imsave(img_path, matrix)


if __name__ == "__main__":
    # # DoS_attacks-Hulk、Bot
    # pcap_path = "./splited_pcap/SQL_Injection/output_1.pcap"  # 替换为你的PCAP文件路径
    # img_output_path = "./newImg/"  # 替换为输出图片的文件夹路径
    # if not os.path.exists(img_output_path):
    #     os.makedirs(img_output_path)
    # packet_encoder(pcap_path, img_output_path)
    source_folder = './new_splited_pcap/Others'
    output_folder = './CIC-IDS-2018-v2/Others'
    
    copy_subdirectories_only_if_not_exist(source_folder, output_folder)
    process_pcap_files(source_folder, output_folder)
