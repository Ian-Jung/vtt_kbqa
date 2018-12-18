# Knowledge Base Question Answering 

### 질문과 질문과 관련된 정답 kb triple이 주어졌을 때 응답 문장을 생성하는 모델
![model](https://user-images.githubusercontent.com/37574306/50127836-17510580-02b6-11e9-9773-1ab4af72af47.png)
##### 질문과 kb triple은 동일 encoder를 통과

##### 필수 라이브러리 : nltk, pytorch
##### 필요 파일 : question file(input1), knowledge base triple file(input2), answer(target) file

##### 각 파일의 입력 예시
| 파일 유형    |     예시    |
| :---------- | :---------: |
| question | who is father of michael jackson |
| kb triple | michael jackson \<p> father \</p> joe jackson |
| answer | joe jackson is father of michael jackson |
> ##### * kb triple에서 relation에 해당하는 단어의 시작과 끝에는 \<p> , \</p>가 각각 마킹되어 있음(선택 사항)
> ##### * 한 쌍의 질문, kb triple, 응답은 각 파일 내에서 동일한 순서로 저장되어 있어야 함

