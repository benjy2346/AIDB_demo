from transformers import pipeline
import sqlite3
from config import *
import numpy as np
from scipy.stats import norm
import random
"""refernece:
Danial Kang. AIDB: Unstructured Data Queries via Fully Virtual Tables
https://drive.google.com/file/d/1XEmDRaNpf5JMjVIOp3VmQvZi2qXyFOhG/view
Barzan Mozafari, Ning Niu. University of Michigan. New Sampling-Based Summary Statistics for Improving Approximate Query Answers 
https://web.eecs.umich.edu/~mozafari/php/data/uploads/approx_chapter.pdf
"""
def read_data(path):
    data = []
    file = open(path, "r")
    i=0
    while True:
        content=file.readline()
        if not content:
            break
        data.append((i,content,i % 3))
        i+=1
    file.close()
    return  data
def create_table():
    connection = sqlite3.connect('AIDB.db')
    cursor = connection.cursor()

    # Create a base class sentense table
    # Source is just the index of sentences mod 3, for testing purpose
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Sentences (
        id INTEGER PRIMARY KEY,
        Sentence TEXT,
        Source INT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Mapping (
        SentenceID INTEGER,
        Label TEXT,
        Score FLOAT,
        FOREIGN KEY (SentenceID) REFERENCES Sentences(id)
    )
    """)
    # Commit the changes
    connection.commit()


def insert_sentences_table(data):
    connection = sqlite3.connect('AIDB.db')
    cursor = connection.cursor()
    
    cursor.executemany("""
    INSERT INTO Sentences (id, Sentence,Source) VALUES (?, ?, ?)
    """, data)
    connection.commit()
    return

def delete_tables():
    connection = sqlite3.connect('AIDB.db')
    cursor = connection.cursor()
    cursor.execute("""
    DROP TABLE IF EXISTS Mapping;
    """)
    cursor.execute("""
    DROP TABLE IF EXISTS Labels;
    """)
    cursor.execute("""
    DROP TABLE IF EXISTS Sentences;
    """)

    # Commit the changes
    connection.commit()

    return

def insert_mapping_table(data):
    connection = sqlite3.connect('AIDB.db')
    cursor = connection.cursor()
    to_push = []
    

    for sentence_id, sentence,source in data:
        
        # process using pipeline sentiment analysis
        results = sentiment_pipeline(sentence)
        for res in results:
            to_push.append((sentence_id,res['label'],res['score']))
    
    cursor.executemany("""
    INSERT INTO Mapping (SentenceID, Label, Score) VALUES (?, ?, ?)
    """, to_push)
    
    connection.commit()
    return

def parse_sql(input_sql:str, sample_size,total_size):
    parts = input_sql.split(' ')
    selected_object = parts[1]
    
    # check if it use AVG, SUM, COUNT methods
    if not '(' in selected_object or not ')' in selected_object:
        raise Exception('not a approximate query, only support AVG, SUM, COUNT')
    
    method = selected_object[:selected_object.find('(')]
    
    obj  = selected_object[selected_object.find('(')+1:selected_object.find(')')]
    
    if "ERROR_TARGET" not in parts or "CONFIDENCE" not in parts:
        raise Exception('Missing ERROR_TARGET or CONFIDENCE parameters')
    error_target = parts[parts.index('ERROR_TARGET')+1]
    confidence = parts[parts.index('CONFIDENCE')+1].replace(';','')
    
    # modify the input sql so that we can randomly choose sample_size many data from the database

    random_id = list(range(total_size))
    random.shuffle(random_id)
    random_id = random_id[:sample_size]
    random_id =[str(id) for id in random_id]
    # we choose sample size many people, and some of them will satisfy the conditions
    input_sql = input_sql.replace("WHERE",'WHERE SentenceID IN ('+", ".join(random_id) + ') AND')
    input_sql = input_sql.replace(selected_object, f'SentenceID, {obj}')
    input_sql = input_sql[:input_sql.find('ERROR_TARGET')]+";" 
        
        
    confidence = int(confidence.replace('%','')) / 100
    error_target = int(error_target.replace('%','')) / 100
    table = parts[parts.index("FROM")+1]
    output = {}
    output['sql'] = input_sql
    output['error_target'] = error_target
    output['confidence'] = confidence   
    output['method'] = method   
    output['obj'] = obj 
    output['table'] = table
    return output
def approximate_query_engine_avg(input_sql,sample_size,total_size, max_looping=20):
    connection = sqlite3.connect('AIDB.db')
    cursor = connection.cursor()
    
    # if the error is too large, increase the sample size and do the query again
    for i in range(max_looping):
        if sample_size >= total_size: break
        parsed = parse_sql(input_sql, sample_size,total_size)  
        
        # obtain query data
        cursor.execute(parsed['sql'])
        results = [data[1] for data in cursor.fetchall()]
        
        mean  = np.mean(results)
        sd = np.std(results)
        
        confidence = parsed['confidence']
        error_target = parsed['error_target']
        z = norm.ppf((1 + confidence) / 2)
        
        # Calculate traditional confidence interval
        traditional_ci_lower = mean - z * (sd / np.sqrt(sample_size))
        traditional_ci_upper = mean + z * (sd / np.sqrt(sample_size))
        
        # Calculate margin of error based on ERROR_TARGET
        target_moe = mean * error_target
        
        # Check if traditional CI meets the ERROR_TARGET criteria
        if z * (sd / np.sqrt(sample_size)) <= target_moe:
            return mean, (traditional_ci_lower,traditional_ci_upper)
        else:
            # a naive method: increase the sample size to meet the ERROR_TARGET criteria.
            sample_size *=2   
    return mean, (traditional_ci_lower,traditional_ci_upper)

def approximate_query_engine_sum(input_sql,sample_size,total_size, max_looping=20):
    connection = sqlite3.connect('AIDB.db')
    cursor = connection.cursor()
    for i in range(max_looping):
        if sample_size >= total_size: break
        parsed = parse_sql(input_sql, sample_size,total_size)  
        cursor.execute(parsed['sql'])

        results = [item[1] for item in cursor.fetchall()]
        
        # apply the formula in reference 1, to get standard deviation and mean
        estimate_sum  = np.sum(results) * total_size /sample_size
        sd = np.sqrt(np.sum([(r-np.mean(results))**2 for r in results]) / total_size)

        confidence = parsed['confidence']
        error_target = parsed['error_target']
        z = norm.ppf((1 + confidence) / 2)
        
        # Calculate traditional confidence interval
        traditional_ci_lower = estimate_sum - z * (sd *total_size)
        traditional_ci_upper = estimate_sum + z * (sd *total_size)
        
        # Calculate margin of error based on ERROR_TARGET
        target_moe = np.mean(results) * error_target
        
        # Check if traditional CI meets the ERROR_TARGET criteria
        if z * (sd / np.sqrt(sample_size)) <= target_moe:
            return estimate_sum, (traditional_ci_lower,traditional_ci_upper)
        else:
            # Increase the sample size to meet the ERROR_TARGET criteria.
            sample_size *=2
            
    return estimate_sum, (traditional_ci_lower,traditional_ci_upper)  

def approximate_query_engine_count(input_sql,sample_size,total_size, max_looping=20):
    connection = sqlite3.connect('AIDB.db')
    cursor = connection.cursor()
    for i in range(max_looping):
        if sample_size >= total_size: break
        parsed = parse_sql(input_sql, sample_size,total_size)  
        cursor.execute(parsed['sql'])

        results = [item[1] for item in cursor.fetchall()]
        
        # apply the formula in reference 1, to get standard deviation and mean
        sample_proportion = len(results) /sample_size
        estimate_count  = len(results) * total_size /sample_size
        sd = np.sqrt(sample_proportion * (1-sample_proportion) /sample_size)
        confidence = parsed['confidence']
        error_target = parsed['error_target']
        z = norm.ppf((1 + confidence) / 2)
        
        # Calculate traditional confidence interval
        traditional_ci_lower = estimate_count - z * (total_size *sd)
        traditional_ci_upper = estimate_count + z * (total_size *sd)
        
        # Calculate margin of error based on ERROR_TARGET
        target_moe = np.mean(results) * error_target
        
        # Check if traditional CI meets the ERROR_TARGET criteria
        if z * (total_size *sd) <= target_moe:
            return estimate_count, (traditional_ci_lower,traditional_ci_upper)
        else:
            # Increase the sample size to meet the ERROR_TARGET criteria.
            sample_size *=2
    return estimate_count, (traditional_ci_lower,traditional_ci_upper)  

def approximate_query_engine(input_sql:str, sample_size,total_size):
    
    # if the error is too large, increase the sample size and do the query again
    parsed = parse_sql(input_sql, sample_size,total_size) 
    if parsed['method'] == "AVG":
        return approximate_query_engine_avg(input_sql, sample_size,total_size)
    elif parsed['method'] == "SUM":
        return approximate_query_engine_sum(input_sql, sample_size,total_size) 
    elif parsed['method'] == 'COUNT':
        return approximate_query_engine_count(input_sql, sample_size,total_size)
    return None
def exact_query_engine(input_sql:str):
    connection = sqlite3.connect('AIDB.db')
    cursor = connection.cursor()
    if "ERROR_TARGET" in input_sql:
        input_sql = input_sql[:input_sql.index("ERROR_TARGET")]+';'
    cursor.execute(input_sql)
    
    return cursor.fetchall()
if __name__ == "__main__":
    
    delete_tables()
    data = read_data(DATA_PATH)
    create_table()
    
    insert_sentences_table(data)
    insert_mapping_table(data)
    
    ''' structure of the mapping table       
    SentencesID INTEGER,
    Label INTEGER,
    Score FLOAT,
    FOREIGN KEY (SentenceID) REFERENCES S entences(id),'''
    sample_size = 40
    total_size = len(data)
    
    # test SUM() approximate query
    input_sql_sum = '''SELECT SUM(score) FROM Mapping WHERE label = 'POSITIVE' ERROR_TARGET 10% CONFIDENCE 95%;'''
    approx_results = approximate_query_engine(input_sql_sum, sample_size,total_size)
    actual_results = exact_query_engine(input_sql_sum)
    print('test SUM() approximate query:')
    print("with sql: "+input_sql_sum)
    print(f"approximate_results of SUM() = {approx_results}")
    print(f"actual_results of SUM() = {actual_results}")
    print()
    
    # test AVG() nested, approximate query:
    input_sql_avg = '''SELECT AVG(score) FROM Mapping WHERE SentenceID IN ( SELECT id FROM Sentences WHERE Source = 1 ) ERROR_TARGET 10% CONFIDENCE 95%;'''
    approx_results = approximate_query_engine(input_sql_avg, sample_size,total_size)
    actual_results = exact_query_engine(input_sql_avg)
    print('test AVG() approximate query and nested query:')
    print("with sql: "+input_sql_avg)
    print(f"approximate_results of AVG() = {approx_results}")
    print(f"actual_results of AVG()= {actual_results}")
    print()
    
    # test COUNT() approximate query
    input_sql_count = '''SELECT COUNT(score) FROM Mapping WHERE label = 'POSITIVE' ERROR_TARGET 10% CONFIDENCE 95%;'''
    approx_results = approximate_query_engine(input_sql_count, sample_size,total_size)
    actual_results = exact_query_engine(input_sql_count)
    print('test COUNT() approximate query:')
    print("with sql: "+input_sql_count)
    print(f"approximate_results of COUNT() = {approx_results}")
    print(f"actual_results of COUNT() = {actual_results}")
    print()