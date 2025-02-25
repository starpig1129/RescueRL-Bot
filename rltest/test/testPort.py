import socket

def tcp_client():
    host = '127.0.0.1'  # 本地主機
    port = 9000  # 目標端口

    # 創建 socket 物件
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # 連接到伺服器
        client_socket.connect((host, port))
        print(f'成功連接到 {host}:{port}')
        
        while True:
            message = input("請輸入要發送的訊息 (輸入 'exit' 退出): ")
            if message.lower() == 'exit':
                break
            
            client_socket.sendall(message.encode('utf-8'))  # 發送數據
            response = client_socket.recv(1024)  # 接收數據
            print(f'伺服器回應: {response.decode("utf-8")}')
    
    except ConnectionRefusedError:
        print("無法連接到伺服器，請確認伺服器是否在運行。")
    except Exception as e:
        print(f'發生錯誤: {e}')
    finally:
        client_socket.close()
        print('連線已關閉')

if __name__ == "__main__":
    tcp_client()