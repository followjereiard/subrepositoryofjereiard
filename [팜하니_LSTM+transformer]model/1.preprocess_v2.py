import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from collections import defaultdict

# ===================== 시계열 처리 함수들 =====================

def parse_dose_string(dose_str):
    """
    DOSE(day) 문자열을 파싱하여 투여 정보를 추출
    예: "Vancomycin 1000mg q12h (3/22 0P02/10P04, 3/23 9A23/9P02)"
    """
    if pd.isna(dose_str) or dose_str == '':
        return []
    
    doses = []
    
    # 각 투여 라인을 분리 (줄바꿈으로 구분)
    dose_lines = dose_str.split('\n')
    
    for line in dose_lines:
        if 'Vancomycin' not in line:
            continue
            
        # 용량 추출 (mg)
        dose_match = re.search(r'(\d+)mg', line)
        dose = int(dose_match.group(1)) if dose_match else 0
        
        # 날짜와 시간 추출
        # 패턴: (3/22 0P02/10P04, 3/23 9A23/9P02)
        time_pattern = r'\(([^)]+)\)'
        time_match = re.search(time_pattern, line)
        
        if time_match:
            time_info = time_match.group(1)
            # 각 날짜별 투여 시간들을 분리
            date_times = time_info.split(', ')
            
            for date_time in date_times:
                date_time = date_time.strip()
                # 날짜 추출 (예: 3/22)
                date_match = re.search(r'(\d+/\d+)', date_time)
                if date_match:
                    date_str = date_match.group(1)
                    doses.append({
                        'date': date_str,
                        'dose': dose,
                        'original_line': line.strip()
                    })
    
    return doses

def assign_dose_day_number(all_doses):
    """
    투여 정보를 날짜 순서대로 정렬하여 1day, 2day, 3day... 순서 할당
    날짜 상관없이 투여 순서만 중요
    """
    # 날짜별로 그룹화하여 고유한 투여일들을 찾기
    unique_dates = sorted(set(dose['date'] for dose in all_doses))
    
    # 각 투여 정보에 day 번호 할당 (1부터 시작)
    date_to_day = {date: idx + 1 for idx, date in enumerate(unique_dates)}
    
    for dose in all_doses:
        dose['day_number'] = date_to_day[dose['date']]
    
    return all_doses, len(unique_dates)

def natural_sort_key(text):
    """
    자연스러운 숫자 정렬을 위한 키 함수
    Day_1, Day_2, ..., Day_10, Day_11 순서로 정렬
    """
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', text)]

def calculate_cumulative_dose(all_doses):
    """
    누적 투여량 계산
    """
    total_dose = sum(dose['dose'] for dose in all_doses)
    return total_dose

def calculate_treatment_days(group):
    """
    총 투약일수 계산 (첫 RequestDate부터 마지막 RequestDate까지)
    """
    if len(group) <= 1:
        return 1
    
    dates = pd.to_datetime(group['RequestDate'])
    first_date = dates.min()
    last_date = dates.max()
    treatment_days = (last_date - first_date).days + 1
    return treatment_days

def calculate_bmi(height_cm, weight_kg):
    """
    BMI 계산
    """
    if pd.isna(height_cm) or pd.isna(weight_kg) or height_cm <= 0 or weight_kg <= 0:
        return None
    
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)

def extract_age_number(age_str):
    """
    나이 문자열에서 숫자 추출 (예: " 90 Y" -> 90)
    """
    if pd.isna(age_str):
        return None
    
    age_match = re.search(r'(\d+)', str(age_str))
    if age_match:
        return int(age_match.group(1))
    return None

def process_single_patient_cycle(patient_id, group, cycle_num, processed_data):
    """
    단일 치료 주기의 환자 데이터를 처리 - 전체 기간을 하나의 시계열로 유지
    """
    # 모든 투여 정보 수집
    all_doses = []
    for idx, row in group.iterrows():
        doses = parse_dose_string(row['DOSE(day)'])
        for dose in doses:
            dose['row_idx'] = idx
            dose['request_date'] = row['RequestDate']
        all_doses.extend(doses)
    
    if not all_doses:
        return
        
    # 투여 순서 할당 (날짜 순서대로 1day, 2day, 3day...)
    all_doses, max_days = assign_dose_day_number(all_doses)
    
    # 추가 계산들
    cumulative_dose = calculate_cumulative_dose(all_doses)
    treatment_days = calculate_treatment_days(group)
    
    # 기본 환자 정보 (마지막 행의 정보 사용)
    last_row = group.iloc[-1]
    
    # PatientID 생성 (cycle 정보만 포함)
    if cycle_num > 1:
        final_patient_id = f"{patient_id}_cycle{cycle_num}"
    else:
        final_patient_id = patient_id
    
    # 나이 숫자 추출
    age_numeric = extract_age_number(last_row['Age'])
    
    # BMI 계산
    height = last_row['Height'] if pd.notna(last_row['Height']) else None
    weight = last_row['Weight'] if pd.notna(last_row['Weight']) else None
    bmi = calculate_bmi(height, weight)
    
    # 환자 분류
    is_pediatric = age_numeric is not None and age_numeric <= 19
    is_obese = bmi is not None and bmi >= 30.0
    is_ckd = pd.notna(last_row['CKD 혹은 HD']) and str(last_row['CKD 혹은 HD']).upper() in ['YES', 'Y', '1', 'TRUE']
    
    # 정상 성인 여부 (비만, 소아, CKD 모두 아닌 경우)
    is_normal_adult = not (is_pediatric or is_obese or is_ckd)
    
    patient_data = {
        'PatientID': final_patient_id,
        'Sex': last_row['Sex'],
        'Age': last_row['Age'],
        'Age_Numeric': age_numeric,
        'Height': height,
        'Weight': weight,
        'BMI': bmi,
        'CKD_or_HD': last_row['CKD 혹은 HD'],
        'MRSA_status': last_row['MRSA 여부'],
        'Comedication': last_row['Comedication'],
        'TDM_Runcount': last_row['TDM_Runcount'],
        
        # 새로 계산된 항목들
        'Cumulative_DOSE': cumulative_dose,
        'Total_Treatment_Days': treatment_days,
        'Total_Days': max_days,  # 실제 투여일수
        'Vancomycin_History_Count': cycle_num,
        'Is_Pediatric': is_pediatric,
        'Is_Obese': is_obese,
        'Is_CKD': is_ckd,
        'Is_Normal_Adult': is_normal_adult
    }
    
    # 투여일별로 정리
    dose_by_day = defaultdict(list)
    for dose in all_doses:
        dose_by_day[dose['day_number']].append(dose['dose'])
    
    # 각 투여일별 용량 추가 (Day_1_Dose ~ Day_N_Dose)
    for day in range(1, max_days + 1):
        if day in dose_by_day:
            total_dose = sum(dose_by_day[day])
            patient_data[f'Day_{day}_Dose'] = total_dose
        else:
            patient_data[f'Day_{day}_Dose'] = 0
    
    # TDM 측정 데이터 매핑
    day_measurements = {}
    for idx, row in group.iterrows():
        # 각 TDM 측정일이 몇 번째 투여일인지 확인
        row_doses = parse_dose_string(row['DOSE(day)'])
        if row_doses:
            # 첫 번째 투여일로 대표
            first_dose_date = row_doses[0]['date']
            for dose in all_doses:
                if dose['date'] == first_dose_date:
                    measurement_day = dose['day_number']
                    day_measurements[measurement_day] = row
                    break
    
    # 시계열 변화하는 변수들 추가
    for day in range(1, max_days + 1):
        if day in day_measurements:
            measurement = day_measurements[day]
            patient_data[f'Weight_Day_{day}'] = measurement['Weight']
            patient_data[f'AUC_Day_{day}'] = measurement['AUC']
            patient_data[f'AUC_MIC_Day_{day}'] = measurement['AUC/MIC']
            patient_data[f'Total_DOSE_Day_{day}'] = measurement['total DOSE/day']
            patient_data[f'VDistLiter_Day_{day}'] = measurement['VDistLiter']
            patient_data[f'Clearence_Day_{day}'] = measurement['ClearenceLiterPerkilogram (L/hr) [PKS]']
            patient_data[f'Halflife_Day_{day}'] = measurement['Halflife']
            patient_data[f'EstimatedPeak_Day_{day}'] = measurement['EstimatedPeak']
            patient_data[f'EstimatedTrough_Day_{day}'] = measurement['EstimatedTrough']
        else:
            # 해당 일자에 측정값이 없으면 None
            patient_data[f'Weight_Day_{day}'] = None
            patient_data[f'AUC_Day_{day}'] = None
            patient_data[f'AUC_MIC_Day_{day}'] = None
            patient_data[f'Total_DOSE_Day_{day}'] = None
            patient_data[f'VDistLiter_Day_{day}'] = None
            patient_data[f'Clearence_Day_{day}'] = None
            patient_data[f'Halflife_Day_{day}'] = None
            patient_data[f'EstimatedPeak_Day_{day}'] = None
            patient_data[f'EstimatedTrough_Day_{day}'] = None
    
    processed_data.append(patient_data)

def process_patient_data(df):
    """
    환자별 데이터를 시계열로 변환
    """
    # 결과를 저장할 리스트
    processed_data = []
    
    # 환자별로 그룹화
    for patient_id, group in df.groupby('PatientID'):
        # 날짜순으로 정렬
        group = group.sort_values('RequestDate').reset_index(drop=True)
        
        # 일주일(7일) 이상 간격으로 분리된 치료 주기를 별도 개체로 처리
        if len(group) > 1:
            # 날짜 간격을 확인하여 치료 주기를 분리
            group['RequestDate_dt'] = pd.to_datetime(group['RequestDate'])
            treatment_cycles = []
            current_cycle = [group.iloc[0]]
            
            for i in range(1, len(group)):
                prev_date = group.iloc[i-1]['RequestDate_dt']
                curr_date = group.iloc[i]['RequestDate_dt']
                
                # 일주일(7일) 이상 차이나면 새로운 치료 주기로 분리
                if (curr_date - prev_date).days > 7:
                    treatment_cycles.append(pd.DataFrame(current_cycle))
                    current_cycle = [group.iloc[i]]
                else:
                    current_cycle.append(group.iloc[i])
            
            # 마지막 치료 주기 추가
            treatment_cycles.append(pd.DataFrame(current_cycle))
            
            # 여러 치료 주기가 있는 경우, 각각을 별도로 처리
            if len(treatment_cycles) > 1:
                for cycle_idx, cycle_group in enumerate(treatment_cycles):
                    cycle_group = cycle_group.reset_index(drop=True)
                    # 각 치료 주기를 별도 환자 ID로 처리 (원본ID_cycle1, 원본ID_cycle2 형태)
                    process_single_patient_cycle(patient_id, cycle_group, cycle_idx + 1, processed_data)
            else:
                # 하나의 치료 주기만 있으면 그대로 진행
                group = treatment_cycles[0].reset_index(drop=True)
                process_single_patient_cycle(patient_id, group, 1, processed_data)
        else:
            # 단일 측정만 있는 경우
            process_single_patient_cycle(patient_id, group, 1, processed_data)
    
    return pd.DataFrame(processed_data)

# ===================== Fungus 데이터 병합 함수들 =====================

def extract_patient_id_from_code(id_code):
    """
    환자 ID를 그대로 반환 (suffix 포함)
    12226797-01 -> 12226797-01 (변경 없음)
    """
    if pd.isna(id_code):
        return None
    return str(id_code).strip()

def parse_culture_result(text):
    """
    동정결과에서 균 종류 추출
    """
    if pd.isna(text) or not isinstance(text, str):
        return None
    
    text = text.lower()
    
    # S.aureus 계열
    if 'staphylococcus aureus' in text:
        return 'S.aureus'
    elif 'staphylococcus' in text and 'aureus' in text:
        return 'S.aureus'
    
    # Enterococcus 계열
    elif 'enterococcus' in text:
        return 'Enterococcus'
    
    # 기타 균
    elif '동정결과' in text and ('no' not in text.lower() and 'isolated' not in text.lower()):
        return 'Other_bacteria'
    
    return None

def find_culture_start_date(culture_results):
    """
    배양 결과에서 가장 빠른 동정결과 날짜 찾기
    """
    if not culture_results:
        return None
    
    # 동정결과가 있는 항목들의 날짜를 찾음 (추후 구현)
    # 현재는 첫 번째 결과의 날짜 반환
    return None

def process_fungus_data(fungus_file_path):
    """
    fungus_data.csv (원래 TDM2.xlsx) 파일 처리
    """
    print("Fungus 데이터 파일을 읽는 중...")
    
    try:
        # CSV 파일 읽기
        df_fungus = pd.read_csv(fungus_file_path, encoding='utf-8-sig')
        print(f"Fungus 데이터 로드 완료: {len(df_fungus)}행")
        
    except UnicodeDecodeError:
        # 다른 인코딩 시도
        df_fungus = pd.read_csv(fungus_file_path, encoding='cp949')
        print(f"Fungus 데이터 로드 완료 (cp949): {len(df_fungus)}행")
    
    # 컬럼명 확인 및 정리
    print("컬럼 정보:", df_fungus.columns.tolist()[:10])
    
    # 환자별 데이터 정리
    patient_fungus_data = {}
    
    # 각 행을 순회하며 환자별로 데이터 수집
    for idx, row in df_fungus.iterrows():
        # B열에서 환자 ID 추출 (인덱스 1)
        patient_code = row.iloc[1] if len(row) > 1 else None
        patient_id = extract_patient_id_from_code(patient_code)
        
        if not patient_id:
            continue
            
        # A열에서 데이터 타입 확인 (인덱스 0)
        data_type = row.iloc[0] if len(row) > 0 else None
        
        if patient_id not in patient_fungus_data:
            patient_fungus_data[patient_id] = {
                'bun_data': [],
                'cr_data': [],
                'culture_results': [],
                'start_date': None,
                'last_date': None,
                'culture_start_date': None
            }
        
        # E열(인덱스 4)에서 시작 날짜 추출
        if len(row) > 4 and pd.notna(row.iloc[4]):
            start_date = pd.to_datetime(row.iloc[4], errors='coerce')
            if start_date and patient_fungus_data[patient_id]['start_date'] is None:
                patient_fungus_data[patient_id]['start_date'] = start_date
        
        # G열(인덱스 6)에서 마지막 날짜 추출
        if len(row) > 6 and pd.notna(row.iloc[6]):
            last_date = pd.to_datetime(row.iloc[6], errors='coerce')
            if last_date:
                current_last = patient_fungus_data[patient_id]['last_date']
                if current_last is None or last_date > current_last:
                    patient_fungus_data[patient_id]['last_date'] = last_date
        
        # 데이터 타입별 처리
        if data_type == 'BUN':
            # I열부터 시계열 BUN 데이터 추출
            bun_values = []
            for col_idx in range(8, len(row)):  # I열(인덱스 8)부터
                if pd.notna(row.iloc[col_idx]) and isinstance(row.iloc[col_idx], (int, float)):
                    bun_values.append(row.iloc[col_idx])
            patient_fungus_data[patient_id]['bun_data'].extend(bun_values)
            
        elif data_type == 'Cr':
            # I열부터 시계열 Cr 데이터 추출
            cr_values = []
            for col_idx in range(8, len(row)):  # I열(인덱스 8)부터
                if pd.notna(row.iloc[col_idx]) and isinstance(row.iloc[col_idx], (int, float)):
                    cr_values.append(row.iloc[col_idx])
            patient_fungus_data[patient_id]['cr_data'].extend(cr_values)
            
        elif data_type in ['RESPI.C', 'GASTRO.C', 'URINARY.C', 'BLOOD.C', 'Other.C']:
            # 동정결과 검색
            for col_idx in range(8, len(row)):  # I열부터
                cell_value = row.iloc[col_idx]
                if pd.notna(cell_value) and isinstance(cell_value, str):
                    bacteria_type = parse_culture_result(cell_value)
                    if bacteria_type:
                        # 동정결과가 발견된 경우, culture_start_date 업데이트
                        if patient_fungus_data[patient_id]['culture_start_date'] is None:
                            # 실제로는 해당 컬럼의 날짜를 계산해야 하지만, 
                            # 현재는 start_date를 사용
                            patient_fungus_data[patient_id]['culture_start_date'] = patient_fungus_data[patient_id]['start_date']
                        
                        patient_fungus_data[patient_id]['culture_results'].append({
                            'type': bacteria_type,
                            'source': data_type,
                            'full_text': cell_value[:100]  # 처음 100자만
                        })
    
    print(f"처리된 환자 수: {len(patient_fungus_data)}명")
    return patient_fungus_data

def merge_with_fungus_data(timeseries_df, patient_fungus_data):
    """
    시계열 데이터와 fungus 데이터 병합
    """
    print("시계열 데이터와 Fungus 데이터 병합 중...")
    
    # 각 환자의 최대 일수 찾기
    max_days_per_patient = {}
    for idx, row in timeseries_df.iterrows():
        patient_id = str(row['PatientID'])
        total_days = row.get('Total_Days', 0)
        if patient_id not in max_days_per_patient or total_days > max_days_per_patient[patient_id]:
            max_days_per_patient[patient_id] = total_days
    
    # 전체 데이터셋의 최대 일수
    global_max_days = max(max_days_per_patient.values()) if max_days_per_patient else 0
    print(f"전체 데이터셋의 최대 치료 일수: {global_max_days}일")
    
    # 미리 모든 BUN/Cr 컬럼을 생성하여 성능 개선
    new_columns = {}
    for day in range(1, global_max_days + 1):
        new_columns[f'BUN_Day_{day}'] = [None] * len(timeseries_df)
        new_columns[f'Cr_Day_{day}'] = [None] * len(timeseries_df)
    
    # 새로운 컬럼들을 한 번에 추가
    new_df = pd.DataFrame(new_columns)
    timeseries_df = pd.concat([timeseries_df, new_df], axis=1)
    
    # 새로운 컬럼들 추가 (각 환자의 실제 일수만큼)
    fungus_base_columns = ['Bacteria_Type', 'Culture_Source', 'Treatment_Duration_Days', 
                          'Culture_Start_Date', 'Treatment_Period_Days']
    
    # 기본 컬럼들을 None으로 초기화
    for col in fungus_base_columns:
        timeseries_df[col] = None
    
    matched_count = 0
    
    # 각 환자별로 데이터 병합
    for idx, row in timeseries_df.iterrows():
        patient_id = str(row['PatientID'])
        patient_days = row.get('Total_Days', 0)
        
        # cycle 정보가 있는 경우 원본 ID 추출
        original_patient_id = patient_id.split('_cycle')[0] if '_cycle' in patient_id else patient_id
        
        # 직접 매칭 시도
        matched = False
        
        # 1. 정확한 매칭 시도
        if original_patient_id in patient_fungus_data:
            fungus_info = patient_fungus_data[original_patient_id]
            matched = True
        # 2. suffix가 있는 경우도 확인 (예: 12226797 -> 12226797-01)
        else:
            for fungus_id in patient_fungus_data.keys():
                if fungus_id.startswith(original_patient_id):
                    fungus_info = patient_fungus_data[fungus_id]
                    matched = True
                    break
        # 3. 반대로 원본 ID에 suffix가 있는 경우 (예: 12226797-01 -> 12226797)
        if not matched and '-' in original_patient_id:
            base_id = original_patient_id.split('-')[0]
            if base_id in patient_fungus_data:
                fungus_info = patient_fungus_data[base_id]
                matched = True
            else:
                # base_id로 시작하는 다른 ID 찾기
                for fungus_id in patient_fungus_data.keys():
                    if fungus_id.startswith(base_id):
                        fungus_info = patient_fungus_data[fungus_id]
                        matched = True
                        break
        
        if matched:
            matched_count += 1
            
            # BUN 데이터
            bun_data = fungus_info['bun_data']
            for day in range(1, min(patient_days + 1, len(bun_data) + 1)):
                timeseries_df.at[idx, f'BUN_Day_{day}'] = bun_data[day-1]
            
            # Cr 데이터
            cr_data = fungus_info['cr_data']
            for day in range(1, min(patient_days + 1, len(cr_data) + 1)):
                timeseries_df.at[idx, f'Cr_Day_{day}'] = cr_data[day-1]
            
            # 균 정보 (우선순위: S.aureus > Enterococcus > Other)
            bacteria_types = [result['type'] for result in fungus_info['culture_results']]
            if 'S.aureus' in bacteria_types:
                timeseries_df.at[idx, 'Bacteria_Type'] = 'S.aureus'
                # S.aureus 결과의 source 찾기
                for result in fungus_info['culture_results']:
                    if result['type'] == 'S.aureus':
                        timeseries_df.at[idx, 'Culture_Source'] = result['source']
                        break
            elif 'Enterococcus' in bacteria_types:
                timeseries_df.at[idx, 'Bacteria_Type'] = 'Enterococcus'
                for result in fungus_info['culture_results']:
                    if result['type'] == 'Enterococcus':
                        timeseries_df.at[idx, 'Culture_Source'] = result['source']
                        break
            elif 'Other_bacteria' in bacteria_types:
                timeseries_df.at[idx, 'Bacteria_Type'] = 'Other_bacteria'
                timeseries_df.at[idx, 'Culture_Source'] = fungus_info['culture_results'][0]['source']
            
            # 치료 기간 계산 (기존)
            if fungus_info['start_date'] and fungus_info['last_date']:
                treatment_duration = (fungus_info['last_date'] - fungus_info['start_date']).days
                timeseries_df.at[idx, 'Treatment_Duration_Days'] = treatment_duration
                timeseries_df.at[idx, 'Culture_Start_Date'] = fungus_info['start_date'].strftime('%Y-%m-%d')
            
            # 새로운 치료기간 계산 (균 검출부터 TDM 종료까지)
            if fungus_info['culture_start_date'] and fungus_info['last_date']:
                treatment_period = (fungus_info['last_date'] - fungus_info['culture_start_date']).days
                timeseries_df.at[idx, 'Treatment_Period_Days'] = treatment_period
    
    print(f"매칭된 환자 수: {matched_count}명 / 전체 {len(timeseries_df)}명")
    
    return timeseries_df

def analyze_patient_groups(df):
    """
    환자군 분석
    """
    print("\n=== 환자군 분석 ===")
    
    # 전체 환자 수
    total_patients = len(df)
    print(f"전체 환자 수: {total_patients}명")
    
    # 환자 분류별 통계
    normal_adult_count = df['Is_Normal_Adult'].sum() if 'Is_Normal_Adult' in df.columns else 0
    pediatric_count = df['Is_Pediatric'].sum() if 'Is_Pediatric' in df.columns else 0
    obese_count = df['Is_Obese'].sum() if 'Is_Obese' in df.columns else 0
    ckd_count = df['Is_CKD'].sum() if 'Is_CKD' in df.columns else 0
    
    print(f"\n환자 분류:")
    print(f"  정상 성인: {normal_adult_count}명 ({normal_adult_count/total_patients*100:.1f}%)")
    print(f"  소아 (≤19세): {pediatric_count}명 ({pediatric_count/total_patients*100:.1f}%)")
    print(f"  비만 (BMI≥30): {obese_count}명 ({obese_count/total_patients*100:.1f}%)")
    print(f"  CKD/HD: {ckd_count}명 ({ckd_count/total_patients*100:.1f}%)")
    
    # BMI 분포
    if 'BMI' in df.columns:
        bmi_data = df['BMI'].dropna()
        if len(bmi_data) > 0:
            print(f"\nBMI 분포:")
            print(f"  평균 BMI: {bmi_data.mean():.1f}")
            print(f"  BMI 범위: {bmi_data.min():.1f} ~ {bmi_data.max():.1f}")
            print(f"  BMI 데이터 있는 환자: {len(bmi_data)}명")
    
    # 나이 분포
    if 'Age_Numeric' in df.columns:
        age_data = df['Age_Numeric'].dropna()
        if len(age_data) > 0:
            print(f"\n나이 분포:")
            print(f"  평균 나이: {age_data.mean():.1f}세")
            print(f"  나이 범위: {age_data.min():.0f} ~ {age_data.max():.0f}세")
    
    # Vancomycin 투약 이력
    if 'Vancomycin_History_Count' in df.columns:
        history_counts = df['Vancomycin_History_Count'].value_counts().sort_index()
        print(f"\nVancomycin 투약 이력:")
        for count, patients in history_counts.items():
            print(f"  {count}회차: {patients}명")
    
    # 치료 기간 분포
    if 'Total_Days' in df.columns:
        days_data = df['Total_Days'].dropna()
        if len(days_data) > 0:
            print(f"\n치료 기간 분포:")
            print(f"  평균 치료일수: {days_data.mean():.1f}일")
            print(f"  최소: {days_data.min():.0f}일, 최대: {days_data.max():.0f}일")
            print(f"  7일 이하: {(days_data <= 7).sum()}명")
            print(f"  8-14일: {((days_data > 7) & (days_data <= 14)).sum()}명")
            print(f"  15-30일: {((days_data > 14) & (days_data <= 30)).sum()}명")
            print(f"  30일 초과: {(days_data > 30).sum()}명")

# ===================== 메인 함수 =====================

def main():
    """
    메인 실행 함수 - 시계열 전처리 + Fungus 데이터 병합 + 모든 계산 항목 추가
    """
    print("=== TDM 데이터 완전 통합 처리 시작 (수정된 시계열 처리) ===\n")
    
    # 1단계: 시계열 전처리
    print("1단계: 시계열 데이터 전처리 및 계산")
    print("-" * 40)
    
    # CSV 파일 읽기
    print("raw_data.csv 파일을 읽는 중...")
    df = pd.read_csv('raw_data.csv')
    
    print(f"원본 데이터: {len(df)}행")
    print(f"고유 환자 수: {df['PatientID'].nunique()}명")
    
    # 중복 환자 및 치료 주기 확인
    duplicate_patients = df[df.duplicated('PatientID', keep=False)]['PatientID'].unique()
    print(f"중복 데이터가 있는 환자 수: {len(duplicate_patients)}명")
    
    # 데이터 변환 (전체 시계열 유지)
    print("\n시계열 데이터 변환 및 계산 중...")
    processed_df = process_patient_data(df)
    print(f"변환된 데이터: {len(processed_df)}행")
    
    # 2단계: Fungus 데이터 병합
    print("\n2단계: Fungus 데이터 병합")
    print("-" * 40)
    
    try:
        # Fungus 데이터 처리
        patient_fungus_data = process_fungus_data('fungus_data.csv')
        
        # 시계열 데이터와 병합
        merged_df = merge_with_fungus_data(processed_df, patient_fungus_data)
        
    except FileNotFoundError:
        print("fungus_data.csv 파일을 찾을 수 없습니다. Fungus 데이터 병합을 건너뜁니다.")
        merged_df = processed_df
        
        # Fungus 컬럼들을 None으로 추가
        fungus_columns = ['Bacteria_Type', 'Culture_Source', 'Treatment_Duration_Days', 
                         'Culture_Start_Date', 'Treatment_Period_Days']
        for col in fungus_columns:
            merged_df[col] = None
    
    # 3단계: 컬럼 정리 및 저장
    print("\n3단계: 최종 결과 정리")
    print("-" * 40)
    
    # 컬럼 순서 정리
    ordered_columns = []
    
    # 1. 기본 환자 정보
    basic_info_cols = ['PatientID', 'Sex', 'Age', 'Age_Numeric', 'Height', 'Weight', 'BMI',
                       'CKD_or_HD', 'MRSA_status', 'Comedication', 'TDM_Runcount']
    ordered_columns.extend([col for col in basic_info_cols if col in merged_df.columns])
    
    # 2. 계산된 항목들
    calculated_cols = ['Cumulative_DOSE', 'Total_Treatment_Days', 'Total_Days', 
                      'Vancomycin_History_Count', 'Is_Pediatric', 'Is_Obese', 'Is_CKD', 'Is_Normal_Adult']
    ordered_columns.extend([col for col in calculated_cols if col in merged_df.columns])
    
    # 3. 균 정보
    bacteria_cols = ['Bacteria_Type', 'Culture_Source', 'Treatment_Duration_Days', 
                    'Culture_Start_Date', 'Treatment_Period_Days']
    ordered_columns.extend([col for col in bacteria_cols if col in merged_df.columns])
    
    # 4. 시계열 데이터 (Day_N 형태의 모든 컬럼)
    day_columns = [col for col in merged_df.columns if 'Day_' in col]
    # 자연스러운 정렬을 위해 숫자 기준으로 정렬
    day_columns_sorted = sorted(day_columns, key=natural_sort_key)
    ordered_columns.extend(day_columns_sorted)
    
    # 중복 제거
    ordered_columns = list(dict.fromkeys(ordered_columns))
    
    # 순서대로 재정렬
    merged_df = merged_df[ordered_columns]
    
    # 최종 파일 저장
    output_filename = 'final_complete_tdm_data_fixed.csv'
    merged_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n최종 완성된 데이터가 '{output_filename}'에 저장되었습니다.")
    
    # 4단계: 결과 요약 및 분석
    print("\n4단계: 최종 결과 요약 및 분석")
    print("-" * 40)
    print(f"총 행 수: {len(merged_df)}명")
    print(f"총 컬럼 수: {len(merged_df.columns)}개")
    
    # 환자군 분석
    analyze_patient_groups(merged_df)
    
    # 분리된 개체 정보 출력
    cycle_patients = len([pid for pid in merged_df['PatientID'] if '_cycle' in str(pid)])
    print(f"\n분리된 개체:")
    print(f"  치료 주기별로 분리된 환자: {cycle_patients}명")
    
    # 균 종류별 분포 확인
    if 'Bacteria_Type' in merged_df.columns:
        bacteria_dist = merged_df['Bacteria_Type'].value_counts()
        print(f"\n균 종류 분포:")
        for bacteria, count in bacteria_dist.items():
            print(f"  {bacteria}: {count}명")
    
    # BUN/Cr 데이터 있는 환자 수
    bun_cols = [col for col in merged_df.columns if col.startswith('BUN_Day_')]
    cr_cols = [col for col in merged_df.columns if col.startswith('Cr_Day_')]
    if bun_cols:
        bun_patients = merged_df[bun_cols[0]].notna().sum()
        print(f"\n신장 기능 데이터:")
        print(f"  BUN 데이터 있는 환자: {bun_patients}명")
    if cr_cols:
        cr_patients = merged_df[cr_cols[0]].notna().sum()
        print(f"  Cr 데이터 있는 환자: {cr_patients}명")
    
    print(f"\n=== 최종 완성! 수정된 시계열 처리 완료 ===")
    
    return merged_df

if __name__ == "__main__":
    # 실행
    result_df = main()