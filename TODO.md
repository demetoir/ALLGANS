# TODO

* find better way to manage todo list like todofy[https://todofy.org/]

* 문서화
  - Docstring 추가
  - ~~파일 트리 및 설명 추가~~
  - 결과물 비교 

* GitHub-Flow 적용
  - 미완성 모델 처리

* ~~데이터 셋 선택 코드 추가~~

* ~~작성자 정보 추가~~

* [fix] 범용 환경에서 작동 가능한 코드로 수정
    - detect tensorboard path in env_setting
        - 사용환경을 자동으로 감지해야함
    - interactive cli for initailize project
        - 처음 실행 할때는 프로그램 설치할떄처럼 CLI 로 instance,data 경로 지정하고 다운로드 하는 스크립트 필요함

* [implement] add more dataset from [http://deeplearning.net/datasets/]

* [implement] implement from old-ALLGANS
    - labeling.py
    - InstanceViewer.py
    - Visualizer.image_conv_filter.py
    - Visualizer.image_interpolation.py
    - Visualizer.log_csv.py
    - Visualizer.image_conv_filter.py
    - model.test_model.BEGAN.py
    - model.test_model.CAE.py
    - model.test_model.multi_stage_gan_type1.py
    - model.test_model.multi_stage_gan_type2.py
    - model.test_model.multi_stage_gan_type3.py
    - model.test_model.multi_stage_gan_type4.py
    - model.test_model.WGAN/*.py
    - data_handler.celebA.py
        - 각 이미지 파일이 따로 jpg로 저장되어있어서 읽어올때 하드디스크 병목이있음
        - 전체 데이터가 커서 모든 데이터가 메모리에 올라가지 못할수있음
    - workbench.celebAHelper.py
    - InstanceManager.InstanceManager.gen_readme()
    - misc todo in source code 
    
- [fix] update tutorial code

- [impement] CLI for InstanceManager, InstanceViewer
    - init data, instace folder path
    - show current managing instance
    - show all instance state
    - auto detect env script
    - set env_setting
    - view Instance Visualizer
    - maybe use some DB for Instance
    - download module in  Dependencies
    - manage dataset

    
* [design] 모든 생성된 instance 관리를 위한 db 설계 필요

* [implemnt] InstanceManager CLI insterface
    - 코드상 말고 consol로 작업실행 하는 코드 필요함

* [fix] fashion mnist 바이트에서 읽도록 수정

* [fix] dataset download script require
    - 적용대상 : cifar10, cifar100, celebA

* [fix] 변경된 api 맞게 tutorial code 수정

* ~~[fix] unit test 폴더는 deprecated 되었음, 수정 또는 삭제~~

* [fix] comment 개선 필요
    - 함수에 대한 인자 및 동작에 대한 설명 필요

* [fix] todo 간소화
- 현재 존재하는 todo중 해결 및 거의 다된것은 제거함
    

* [refactoring] decompose util function and rename
    - ex) numpy related util  must move to util_numpy

* [fix] fix warning message
    - /home/demetoir/anaconda3/envs/tensor/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
  return f(*args, **kwds)


    
    