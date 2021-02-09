주의사항
restart.bat파일의 root를 현 컴퓨터에서 anaconda3가 위치한 폴더로 변경해야 함
restart.bat파일의 activate가 적힌 부분의 뒷부분을 사용하려는 가상환경의 이름으로 변경해야 함
parameter.py내의 addr과 port를 현 컴퓨터의 ip와 사용할 포트로 변경해야 함 (server.py에서 사용)
클라이언트에서 사용하는 addr과 port는 client.cpp파일에서 직접 변수를 수정한 후 다시 컴파일해서 실행파일로 저장해야 함
**서버와 클라이언트 모두 addr의 기본값은 localhost, port의 기본값은 3070으로 설정되어 있다.

서버 실행방법
윈도우 폴더로 이동해 restart.bat실행

클라이언트 실행방법
client.exe파일을 cmd상에서 실행

[파일 설명]
server.py	: 서버 파일
client.py	: 클라이언트 파일
preprocessing.py : 클라이언트에서 사용하는 전처리 함수가 존재하는 파일
parameter.py : 서버, 클라이언트에서 사용할 경로 변수가 저장되어 있는 파일
	json_path -> server에서 사용할 json파일의 경로
	weight_path -> server에서 사용할 학습된 가중치 netG.pth, netD.pth가 존재하는 디렉토리 경로
	addr -> 서버의 사용할 ip주소
	port -> 서버에서 사용할 port번호
restart.bat : 서버 실행, 서버 중단시 다시 서버를 작동시키는 배치 파일
