#include <WinSock2.h>
#include <Ws2tcpip.h>
#include <string>
#include <cstring>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <direct.h>
#pragma comment(lib, "ws2_32.lib")

#define BUFSIZE 1024

using namespace std;

class Client {
public:
    const char* serv_addr;
    short port;
    SOCKADDR_IN target;
    SOCKET sock;

    Client(const char* host = "127.0.0.1", short port_ = 3070) {
        serv_addr = host;
        port = port_;

        target.sin_family = AF_INET;
        target.sin_port = htons(port);
        inet_pton(AF_INET, serv_addr, &target.sin_addr);


        sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    }

    int Connect() {
        if (sock == INVALID_SOCKET)
        {
            cout << "socket() error : " << WSAGetLastError() << endl;
            return -1;
        }

        if (connect(sock, (SOCKADDR*)&target, sizeof(SOCKADDR_IN)) == SOCKET_ERROR) {
            cout << "connect() error : " << WSAGetLastError() << endl;
            return -1;
        }
        cout << "connect success" << endl;
        return 0;
    }

    int send_msg(string input) {
        int n;
        n = send(sock, input.data(), input.size(), 0);
        return n;
    }

    string recv_msg() {
        char buffer[BUFSIZE];
        memset(&buffer, 0, BUFSIZE);
        recv(sock, buffer, BUFSIZE, 0);
        return buffer;
    }

    void close() {
        send_msg("finish");
        closesocket(sock);
    }
};

bool directory_check(string path) {
    struct stat info;


    if (stat(path.c_str(), &info) != 0) {
        cout << "[client]cannot access " << path << endl;

        if (_mkdir(path.c_str()) == 0) {
            cout << "[client]create new dir : " << path << endl;
            return true;
        }
        else {
            cout << "[client]cannot create new dir : " << path << endl;
            return false;
        }
    }
    else if (info.st_mode & S_IFDIR) {
        cout << "[client]path is already exist, don\'t need create new dir" << endl;
        return true;
    }
    else {
        cout << "[client]this is not directory" << endl;
        return false;
    }
}

bool file_check(string path) {
    struct stat info;


    if (stat(path.c_str(), &info) != 0) {
        cout << "cannot access : " << path << endl;
        return false;
    }
    else if (info.st_mode & S_IFREG) {
        cout << "[client]this is file path : " << path << endl;
        return true;
    }
    else {
        cout << "[client]this is not file path : " << path << endl;
        return false;
    }
}


int main() {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        cout << "[client]WSAStartup error" << endl;
        return 0;
    }

    Client C = Client();
    struct stat info;
    string temp_dir_path;
    string temp_file_path;
    string confirm_msg;
    string server_output;
    bool is_dir;

    memset(&info, 0, sizeof(info));

    C.Connect();
    confirm_msg = C.recv_msg();
    cout << "[client]from server : " << confirm_msg << endl;

    while (true) {
        cout << "[client]enter absolute img file path : ";
        cin >> temp_file_path;
        if (temp_dir_path == "q" || temp_dir_path == "quit") {
            C.close();
            WSACleanup();
            return 0;
        }

        if (!file_check(temp_file_path)) {
            continue;
        }
        cout << temp_file_path << endl;
        C.send_msg(temp_file_path);
        server_output = C.recv_msg();
        
        cout << server_output << endl;
    }

    WSACleanup();
    return 0;
}