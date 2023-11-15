import os
import json
list_dir = os.listdir('./essays')

sub_dict = dict()

level =  ['상','중','하']
type_info = {'찬성반대':'논술형', '주장':'논술형','설명글':'수필형','대안제시':'논술형','글짓기': '수필형'}


for file in list_dir:
    with open('./essays/'+file,'r') as f:
        data = json.load(f)
    f.close()
    sub = data['info']['essay_main_subject']
    prompt = data['info']['essay_prompt']
    essay_level = int(data['info']['essay_level'])
    detailed_type = data['info']['essay_type']
    essay_type = type_info[detailed_type]
    sub_dict[sub] = {'prompt':prompt, 
                     'essay_level':level[essay_level-1],
                     'essay_type': essay_type,
                    'detailed_type': detailed_type
                     }
    
with open('./data/subjects.json', 'w') as f:
    json.dump(sub_dict,f,ensure_ascii=False,indent='\t')
f.close()
    
scoring = { 0:['문법의 정확성',
               '문법의 정확성을 평가',
               None],
            1: ['단어 사용의 적절성',
                '상황에 맞는 단어 사용',
                None],
            2: ['문장 표현의 적절성',
                '문장 구조의 다양성 평가 및 문장 길이의 적절성 평가',
                None],
            3: ['문단 간 구조의 적절성',
                '서론, 본론, 결론의 삼단구성 여부 평가 및 각 부분 분량의 적절함',
                None],
            4: ['문단 내 구조의 적절성',
                '주제문장과 보조문장 간 연결성 평가',
                None],
            5: ['구조의 일관성',
                '문장의 연결 관계 및 흐름을 평가',
                None],
            6: ['분량의 적절성',
                '글의 분량이 제시한 분량에 적절한지 평가',
                None],
            7: ['주제의 명료성',
                '글의 주제가 명료하게 드러나는지, 글 전체가 하나의 주제와 관련되어 있는지 평가',
                None],
            8: ['사고의 창의성',
                '문제를 통찰하는데 특이하며, 논리적임을 평가, 새로운 발상이나 관점 전환을 시도했는지 평가',
                None],
            9: ['프롬프트 독해력',
                '프롬프트에 대한 이해를 바탕으로 글을 작성했는지 평가',
                None],
            10: ['설명의 구체성',
                 '설명이 구체적이고 상세한지, 주제 설명이 다양하게 제시되었는지 평가',
                 None]
}
with open('./data/essay_rubric.json', 'w') as f:
    json.dump(scoring, f,ensure_ascii=False,indent='\t')
f.close()
