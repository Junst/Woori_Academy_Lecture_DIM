CIFAR-10을 돌리는 CNN의 Input Image를 내가 만들고 싶다!

나만의 Custom Dataset을 가져와서 만든다!

참고자료 : https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=cenodim&logNo=220946688251

# 원리
CIFAR-10 예제에 사용되는 Binaray 파일을 만든 후, input 데이터의 위치를 CIFAR-10 Dataset에서 내가 만든 Binaray File 위치로 바꾸면 된다.

해당 데이터는 Google Driver crawling을 통해 사진을 저장한 후, 32x32 size로 resize해서 cnn에 넣고 훈련을 시킨다.

