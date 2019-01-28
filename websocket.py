import time
import threading
from websocket_server import WebsocketServer

class Websocket(threading.Thread):
    def __init__(self, port, servers, new_client, client_left, message_received):
        super(Websocket, self).__init__()
        self.port = port
        self.servers = servers
        self.new_client = new_client
        self.client_left = client_left
        self.message_received = message_received

    def run(self):
        server = WebsocketServer(self.port, host="0.0.0.0")
        server.set_fn_new_client(self.new_client)
        server.set_fn_client_left(self.client_left)
        server.set_fn_message_received(self.message_received)
        self.servers.append(server)
        server.run_forever()

if __name__ == '__main__':
    websocket_server = Websocket(9000, servers=[])
    websocket_server.start()
    print(">>>>")
    index = 1
    while True:
        time.sleep(5)
        print(">>>>>>" + str(index))
        if websocket_server.servers is not None:
            websocket_server.servers[0].send_message_to_all("index" + str(index))
        index = index + 1
