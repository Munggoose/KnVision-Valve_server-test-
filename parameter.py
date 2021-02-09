'''
json_path : 모델에 대한 정보를 담고  있는 json file 경로

현재 아래와 같이 구성됨
mod : 'ganomaly'
ganomaly : {
    extralayers : 1,
    nz : 400,
    isize : 256,
    weight_path :  ".\\ganomaly\\output\\ganomaly\\none\\train\\weights"
}


weight_path : json파일이 json_path에 존재하지 않을 때, 사용할 기본 fanomaly 모델의 가중치 경로

addr : 서버의 ip주소
port : 서버가 사용할 port번호


'''
json_path = '.\\json\\serv_json.json'
weight_path = '.\\weights'
addr = '127.0.0.1'
port = 3070