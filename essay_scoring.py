from transformers import TFAutoModel, AutoTokenizer
from kiwipiepy import Kiwi
import numpy as np
import json
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from prettytable import PrettyTable

def tokenize(sentences, tokenizer):
    sent_max_len = 50
    inputs = tokenizer.batch_encode_plus(sentences)
    ids_new = pad_sequences(inputs['input_ids'],maxlen=sent_max_len,padding='post')
    mask_new = pad_sequences(inputs['attention_mask'], maxlen = sent_max_len, padding='post')
    return ids_new , mask_new

def essay_to_sentences(essay_v, essay_topic):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    kiwi = Kiwi()
    sents = kiwi.split_into_sents(essay_v)
    raw_sentences = [sent.text for sent in sents]
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences = essay_topic + '[SEP]' + raw_sentence
            if tokenized_sentences != []:
                sentences.append(tokenized_sentences)
    return sentences

#total score를 얻기위한 weights
def get_weight(subject):
    rubric = pd.read_csv('./data/rubric.csv',encoding='utf-8-sig')
    rubric = (rubric[rubric['SUBJECT'] == subject].values)[0]
    rubric_weights = rubric[[6,7,8,10,11,12,13,15,16,17,18]]
    major_weights = rubric[[5,9,14]]
    not_include = np.where(rubric_weights==0)[0]
    return rubric_weights, major_weights, not_include

def scoring(subject, contents,model, tokenizer):
    # total score를 얻기위한 weights
    rubric_weights , major_weights, not_include = get_weight(subject) 
    
    #글 원문을 문장 리스트로 변환
    sentences = essay_to_sentences(contents,subject)

    #tokenizing
    input_ids ,attention_mask = tokenize(sentences,tokenizer)

    #글 원문의 embedding값 추출
    predict = model(input_ids = input_ids, attention_mask= attention_mask)
    embedding = predict[0].cpu()[:,0,:].numpy()
    emb = [embedding]
    emb = pad_sequences(emb, maxlen=128,padding='pre',dtype='float')

    #임베딩 값을 gru 모델에 입력, 점수 예측
    gru_model = load_model('./data/kobert_model.h5')
    predict = gru_model(emb)
    score = np.round(predict[0]*3)

    #total score 계산
    with open('./data/essay_rubric.json','r') as f:
        essay_rubric = json.load(f)
    f.close()
    weighted_score = score * rubric_weights
    major_score = [np.sum(weighted_score[:3])/np.sum(rubric_weights[:3]),
                   np.sum(weighted_score[3:7])/np.sum(rubric_weights[3:7]),
                   np.sum(weighted_score[7:])/np.sum(rubric_weights[7:])
                   ]
    major_score = np.array(major_score)
    total_score = np.round(((np.sum(major_score*major_weights)/10)/3)*100,2)

    #rubric 가중치가 0이면 점수계산에 포함시키지 않음
    score[not_include] = None

    #dictionary에 결과 정리
    for i in range(len(essay_rubric)):
        essay_rubric[str(i)][2] = float(score[i])

    #json 파일로 결과 저장
    with open('./data/score_result.json','w') as f:
        json.dump(essay_rubric,f,ensure_ascii=False,indent='\t')
    f.close()

    return essay_rubric, total_score

def print_result(essay_rubric):
    table = PrettyTable(['평가 기준', '세부 평가 기준', '평가 점수'])
    for value in essay_rubric.values():
        table.add_row([value[0],value[1],value[2]])
    print(table)


def print_prompt(subjects, subject):
    print('level({0}), type({1}) , detailed_type({2})'.format(subjects[subject]['essay_level'], 
                                                                subjects[subject]['essay_type'], 
                                                                subjects[subject]['detailed_type']
                                                                ))
    print()
    print('prompt: ', subjects[subject]['prompt'].replace('\n',''))
    print()

if __name__ == '__main__':    
    #model 로딩
    model = TFAutoModel.from_pretrained('./model/tfmodel.h5')
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    with open('./data/subjects.json','r') as f:
        subjects = json.load(f)
    f.close()

    while True:
        Input = int(input('(0 or 1 or 2): '))
        if Input == 0:
            dataset = pd.read_csv('./data/dataset.csv',encoding='utf-8-sig')
            test_ids = pd.read_csv('./data/testset.csv',encoding='cp949')
            test_ids_list = test_ids['ID'].to_list()
            dataset = dataset.iloc[test_ids_list]

            essay_idx = int(input('INPUT ID(0~{0}): '.format(len(dataset))))
            contents = dataset['ESSAY_CONTENT'].iloc[essay_idx].replace('<span>', '').replace(
                '</span>', '').replace('\n', '').replace('\t', '').replace('#@문장구분#','')
            subject = dataset['ESSAY_SUBJECT'].iloc[essay_idx]
            labels = np.round(dataset.iloc[essay_idx,13:].to_list())
            print_prompt(subjects,subject)
            print('contents: ', contents)
            print(labels)
        elif Input == 1:
            table = PrettyTable(['글 주제', '주제 ID'])
            sub_list = list(subjects.keys())
            for i in range(len(sub_list)):
                table.add_row([sub_list[i], i])
            print(table)
            sub_id = int(input('작성하고 싶은 글의 주제를 선택해 주세요 (0~49) '))
            subject = sub_list[sub_id]
            print_prompt(subjects,subject)
            contents = input('contents: ')
        else:
            break
        essay_rubric, total_score = scoring(subject,contents,model,tokenizer)
        print_result(essay_rubric)
        print('총 점', total_score)
