from konlpy.tag import Kkma
from konlpy.utils import pprint
import numpy as np
def text_split(sentence):
    kkma = Kkma()
    return (kkma.nouns(sentence))

def find_problem(text):
    problem = ["default","love","relation","health_","study","wealth","job"]
    problem_list = [0,0,0,0,0,0,0]
    default = ["몰라","없어"]
    love = ["연애","남자친구","여자친구","사랑","남친","여친","자기","썸","영화","밥","남사친","여사친","재회","새로운","이별","헤어짐","바람"]
    relation = ["친구","만남","주변인","지인","사람","손절","주변 사람","친우","대화","학교","연락","대학","동네친구","배려","예의","선","베프","친한","인스타","카톡"]
    health = ["건강","병원","만수무강","질병","부상","회복","나이","피곤해","졸려","잠","지쳤다","눈","피곤","피곤하다","졸립다","잠온다"]
    study = ["공부","학업","문제","등수","전교","등급","과목","학점","머리","두뇌","수학","영어","국어","필기","시험","장학금","입시"]
    wealth = ["돈","머니","형편","재산","부동산","지갑","지폐","가사","대박","부도","동전","저금","저축"]
    job = ["직장","공채","취준","취업","취직","직업","직종","채용","면접","자소서","기업","회사","진로","꿈","야망","하다"]
    for item in text:
        if any(str in item for str in default):
            problem_list[0]=problem_list[0]+1
        if any(str in item for str in love):
            problem_list[1]=problem_list[1]+1
        if any(str in item for str in relation):
            problem_list[2]=problem_list[2]+1
        if any(str in item for str in health):
            problem_list[3]=problem_list[3]+1
        if any(str in item for str in study):
            problem_list[4]=problem_list[4]+1
        if any(str in item for str in wealth):
            problem_list[5]=problem_list[5]+1
        if any(str in item for str in job):
            problem_list[6]=problem_list[6]+1   

    return problem[np.argmax(problem_list)]

#사용 예시
# from find_problem import find_problem
# find_problem("나는 요즘 너무 형편이 어려워. 이제 더이상 부모님의 수입이 없거든. 알바하면서 내 자취방 돈을 벌어야해. 이제 부족한 돈을 어떻게 처리해야할지 모르겠어")