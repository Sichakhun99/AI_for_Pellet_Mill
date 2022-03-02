import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datetime import *
from pycaret.regression import *
from pycaret.utils import check_metric


def retrieve_data():
    # **** Get PL start record number **** #
    pl_config = pd.read_excel('pl_data/pl_config.xlsx')
    # **** Assign Line **** #
    pl_line = [3]
    # pl_line = range(1,6)

    for pl in pl_line:
        print(f'Retrieve data PL{pl} start')
        # **** Create list to contain data **** #
        data = []
        # **** Define record number to start query **** #
        start_at = pl_config.loc[0, f'PL_{pl}']
        record_start = pl_config.loc[0, f'PL_{pl}']
        # **** Create return data detector **** #
        number_return_record = 1

        while number_return_record > 0:

            # **** Reset detector **** #
            number_return_record = 0
            # **** API **** #
            url = f'http://10.4.10.248:29201/apiPellet/WebService1.asmx/Call_Pellet_data?Pellet_Number={pl}&Formula_Code=&Time_Set=ALLDAY&Limits=100&Start_as={start_at}'
            response = requests.get(url)
            # **** Extract XML **** #
            root = ET.fromstring(response.text)

            xml = False
            for tags in root.iter('Table1'):
                xml = True
                # **** Set Dictionary of obtain data **** #
                transaction = {}
                for t in tags:
                    # **** Convert data to transaction **** #
                    transaction[t.tag] = t.text
                # **** Collect transaction **** #
                data.append(transaction)
                if len(transaction):
                    # **** Count return record **** #
                    number_return_record += 1
            if xml:
                # **** Set record number to start next time **** #
                start_at = int(transaction['id'])

        # **** Save PL Config **** #
        pl_config.loc[0, f'PL_{pl}'] = start_at
        pl_config.to_excel(f'pl_data/pl_config.xlsx', index=False)
        print(f'Retrieve data PL{pl} stop')
        print(f'Number record : {start_at - record_start}')
        if len(data):
            print(f'Data PL{pl} saving')
            # **** Load data of PL **** #
            pl_data = pd.read_excel(f'pl_data/pl{pl}.xlsx')
            # **** Create dataframe for this retrieve **** #
            df_temp = pd.DataFrame(data)
            # **** Append temp dataframe to master **** #
            pl_data = pl_data.append(df_temp, ignore_index=True)
            # **** Save data PL **** #
            pl_data.to_excel(f'pl_data/pl{pl}.xlsx', index=False)
            print(f'Data PL{pl} success')
        else:
            print(f'*** Not found new record of PL{pl} from server')

    return


def ETL():
    # **** Assign Line **** #
    pl_line = [3]
    # pl_line = range(1,6)

    for pl in pl_line:
        print(f'ETL PL{pl} Start')
        # **** Get data raw and parse date **** #
        df = pd.read_excel(f"pl_data/pl{pl}.xlsx")
        # **** Sort ID **** #
        df.sort_values("id", inplace=True)
        df['formular_date'] = df['formular_date'].astype(str)
        # **** Transform formular date to category **** #
        df['formular_date'] = [datetime.strptime(
            t, "%d/%m/%y") for t in df.formular_date]
        df["formular_month"] = [i.month for i in df.formular_date]
        df.drop("formular_date", axis=1, inplace=True)

        # **** Filter steady state **** #
        df.drop(df[(df.motor_power != 1)].index, inplace=True)
        df.drop(df[(df.landing_status != 0)].index, inplace=True)
        df.drop(df[(df.step_finished != 8)].index, inplace=True)
        df.drop(df[(df.steam_valve_act != 1)].index, inplace=True)

        # **** Assign formular group **** #
        group = pd.read_excel("pl_cleaned/FeedGroup.xlsx")
        df = pd.merge(
            df, group[["formular_group", "formular_code"]], on="formular_code", how="left")
        # **** Check incorrect formular code ***** #
        uncode = df[df.formular_code == 0]["formular_name"].drop_duplicates()
        # CASE: FIND INCORRECT
        if len(uncode):
            print('System Alarm : Found Formula Code = 0')
            # **** Get formular name **** #
            name_uncode = []
            for n in uncode:
                name_uncode.append(n.strip().replace(" ", ""))
            # **** Find formular code by formular name in master **** #
            if len(name_uncode):
                code_list = []
                for n in name_uncode:
                    for i, c in enumerate(group.formular_name):
                        if n == c.strip().replace(" ", ""):
                            code_list.append(group.loc[i, "formular_code"])
                # **** Assign formular code in dataframe **** #
                if len(code_list):
                    uncode = df[df.formular_code == 0]
                    for i in range(len(uncode)):
                        df.loc[uncode.index[i], "formular_code"] = code_list[1]
        # CASE: NEW FORMULAR CODE
        ungroup = df[df.formular_group.isna()][["formular_code",
                                                "formular_name"]].drop_duplicates()
        # **** Save new formular code in master **** #
        if len(ungroup):
            print('System Alarm : Found New Formular Code')
            group = pd.concat([group, ungroup], ignore_index=True)
            group.drop_duplicates(inplace=True)
            group.to_excel("pl_cleaned/FeedGroup.xlsx", index=False)

        # **** Define necessary columns **** #
        necessary = [
            "pl_name",
            # "job_id",
            "formular_code",
            "formular_name",
            "fat_target_percent",
            "molass_target_percent",
            "press_legth_die",
            "level_die",
            "ton_die",
            "ton_roller",
            "persen_feed",
            "persen_feed_target",
            "persen_valve",
            "persen_valve_target",
            "bar_steam",
            "temp_hot",
            "temp_cond_target",
            "temp_cond",
            "steam_kg_ton",
            "steam_kg_ton_target",
            "steam_quality",
            "ton_per_hr",
            "ton_per_hr_target",
            "amp_motor",
            "max_amp_motor",
            "fpqf",
            "density",
            # "time_stamp",
            "operate_mode",
            "formular_season",
            "formular_month",
            "formular_group",
        ]
        # **** Drop unuse columns **** #
        [df.drop(col, axis=1, inplace=True)
         for col in df.columns if col not in necessary]
        # **** Format data type **** #
        df["formular_code"] = df["formular_code"].astype(str)
        df[["amp_motor", "temp_cond_target"]] = df[[
            "amp_motor", "temp_cond_target"]].astype(int)
        cols = [
            "persen_feed",
            "persen_valve",
            "bar_steam",
            "temp_hot",
            "temp_cond",
            "steam_kg_ton",
            "ton_per_hr",
            "ton_per_hr_target",
            "persen_feed_target",
            "persen_valve_target",
            "steam_kg_ton_target",
        ]
        for c in cols:
            df[c] = df[c].round(1).astype(float)

        # **** Remove outlier **** #
        features = ["amp_motor", "ton_per_hr", "steam_kg_ton"]
        for feature in features:
            for code in df["formular_code"].unique():
                # **** Filter by fromular code **** #
                selected = df[df.formular_code == code]
                # **** Define Q1 **** #
                q1 = np.percentile(
                    selected[feature], 25, interpolation="midpoint")
                # **** Define Q3 **** #
                q3 = np.percentile(
                    selected[feature], 75, interpolation="midpoint")
                # **** Define IQR **** #
                iqr = q3 - q1
                # **** Define Outlier **** #
                max_boundary = q3 + (1.5 * iqr)
                min_boundary = q1 - (1.5 * iqr)
                # **** Drop outlier **** #
                index_list = selected[(selected[feature] < min_boundary) | (
                    selected[feature] > max_boundary)].index
                df.drop(index_list, inplace=True)
                index_list = []
        # **** Save Cleaned data **** #
        df.to_excel(f"pl_cleaned/cleaned_pl{pl}.xlsx", index=False)
        print(f'ETL PL{pl} Complete')
    return


def steam_ratio_model(pl, data, fraction):
    # **** Define Use Columns **** #
    steam_use = [
        "pl_name",
        "formular_code",
        # "press_legth_die",
        # "level_die",
        # "ton_die",
        # "ton_roller",
        "bar_steam",
        "steam_kg_ton",
        "ton_per_hr",
        "fpqf",
        "density",
        "fat_target_percent",
        "molass_target_percent",
        "formular_month",
        "formular_group",
    ]

    # **** Create dataset for machine learning **** #
    steam_dataset = data[steam_use].copy()
    # **** Create train test dataset **** #
    data_train = steam_dataset.sample(frac=fraction, random_state=123)
    # **** Create unseen dataset **** #
    data_unseen = steam_dataset.drop(data_train.index)
    # **** Reset Index **** #
    data_train.reset_index(drop=True, inplace=True)
    data_unseen.reset_index(drop=True, inplace=True)

    MODEL = f"Model_PL{pl}_Steam_Ratio"
    # **** Load Exist Model **** #
    steam_model_exist = load_model(MODEL, verbose=False)
    # **** Predict Unseen data **** #
    predictions = predict_model(steam_model_exist, data=data_unseen)
    # **** Get exist model score **** #
    exist_model_score = check_metric(
        predictions.steam_kg_ton, predictions.Label, "R2")
    # **** Set accuracy score **** #
    accuracy_score = exist_model_score

    # **** Create Pipeline **** #
    print('*'*40)
    print(f'Generate Model Steam Ratio for PL{pl}')
    print('*'*40)

    steam_regression = setup(
        data_train,
        # **** Define Label **** #
        target="steam_kg_ton",
        # **** Define Numeric Features **** #
        numeric_features=["fat_target_percent", "molass_target_percent"],
        # **** Define Categorical Features **** #
        categorical_features=["formular_code"],
        # **** Standardization **** #
        normalize=True,
        transformation=True,
        combine_rare_levels=True,
        rare_level_threshold=0.05,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        log_experiment=True,
        experiment_name="log_ex",
        session_id=123,
        verbose=False
    )
    # **** Compare Models and get top r2 score from train dataset **** #
    steam_model = compare_models(verbose=False)
    # **** Predict model from test dataset **** #
    predict_model(steam_model, verbose=False)
    # **** Get model r2 score **** #
    results = pull()
    steam_model_r2 = results.R2[0]
    # **** Tuning hyper parameter and create model from train dataset **** #
    steam_model_tuned = tune_model(steam_model, verbose=False)
    # **** Predict tuned model from test dataset **** #
    predict_model(steam_model_tuned, verbose=False)
    # **** Get tuned model r2 score **** #
    results = pull()
    tuned_steam_model_r2 = results.R2[0]
    # **** Select best model score **** #
    if tuned_steam_model_r2 > steam_model_r2:
        steam_model = steam_model_tuned
    # **** Create final model by train test dataset **** #
    steam_model_final = finalize_model(steam_model)
    # **** Get Model Name **** #
    results = pull()
    algorithm_name = results.Model[0]
    # **** Predict final model from unseen dataset **** #
    predictions = predict_model(steam_model_final, data=data_unseen)
    # **** Accuracy Score **** #
    new_model_score = check_metric(
        predictions.steam_kg_ton, predictions.Label, "R2")

    # **** Load log models **** #
    log = pd.read_excel("pl_models/model_log.xlsx")
    index = log[log.Model_Name == MODEL].index

    if new_model_score > exist_model_score:
        # **** Save Model **** #
        save_model(steam_model_final, MODEL, verbose=False)
        # **** Assing model algorithm name **** #
        log.loc[index, "Algorithm"] = algorithm_name
        # **** Assing model accuracy score **** #
        log.loc[index, "Accuracy"] = new_model_score
        # **** Assing model create date **** #
        log.loc[index, "Version"] = pd.Timestamp.today().strftime("%Y-%m-%d")
        # **** Set accuracy score **** #
        accuracy_score = new_model_score
    else:
        log.loc[index, "Accuracy"] = exist_model_score
    log.to_excel("pl_models/model_log.xlsx", index=False)

    return algorithm_name, accuracy_score


def amp_motor_model(pl, data, fraction):
    # **** Define Use Columns **** #
    amp_use = [
        "pl_name",
        "formular_code",
        # "press_legth_die",
        # "level_die",
        # "ton_die",
        # "ton_roller",
        "bar_steam",
        "steam_kg_ton",
        "ton_per_hr",
        "fpqf",
        "density",
        "fat_target_percent",
        "molass_target_percent",
        "formular_month",
        "formular_group",
        "amp_motor",
    ]
    # **** Create dataset for machine learning **** #
    amp_dataset = data[amp_use].copy()
    # **** Create train test dataset **** #
    data_train = amp_dataset.sample(frac=fraction, random_state=123)
    # **** Create unseen dataset **** #
    data_unseen = amp_dataset.drop(data_train.index)
    # **** Reset index **** #
    data_train.reset_index(drop=True, inplace=True)
    data_unseen.reset_index(drop=True, inplace=True)

    MODEL = f"Model_PL{pl}_Amp_Motor"

    # **** Load Exist Model **** #
    amp_model_exist = load_model(MODEL, verbose=False)
    # **** Predict Unseen data **** #
    predictions = predict_model(amp_model_exist, data=data_unseen)
    # **** Get exist model score **** #
    exist_model_score = check_metric(
        predictions.amp_motor, predictions.Label, "R2")
    # **** Set accuracy score **** #
    accuracy_score = exist_model_score

    # **** Create Pipeline **** #
    print('*'*40)
    print(f'Generate Model Amp Motor for PL{pl}')
    print('*'*40)
    amp_regression = setup(
        data_train,
        # **** Define Label **** #
        target="amp_motor",
        # **** Define Numeric Features **** #
        numeric_features=["fat_target_percent", "molass_target_percent"],
        # **** Define Categorical Features **** #
        categorical_features=["formular_code"],
        # **** Standardization **** #
        normalize=True,
        transformation=True,
        combine_rare_levels=True,
        rare_level_threshold=0.05,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        log_experiment=True,
        experiment_name="log_ex",
        session_id=123,
        verbose=False
    )

    # **** Compare Models and get top r2 score from train dataset **** #
    amp_model = compare_models(verbose=False)
    # **** Predict model from test dataset **** #
    predict_model(amp_model, verbose=False)
    # **** Get model r2 score **** #
    results = pull()
    amp_model_r2 = results.R2[0]
    # **** Tuning hyper parameter and create model from train dataset **** #
    amp_model_tuned = tune_model(amp_model, verbose=False)
    # **** Predict tuned model from test dataset **** #
    predict_model(amp_model_tuned, verbose=False)
    # **** Get tuned model r2 score **** #
    results = pull()
    tuned_amp_model_r2 = results.R2[0]
    # **** Select best model score **** #
    if tuned_amp_model_r2 > amp_model_r2:
        amp_model = amp_model_tuned
    # **** Create final model by train test dataset **** #
    amp_model_final = finalize_model(amp_model)
    # **** Get Model Name **** #
    results = pull()
    algorithm_name = results.Model[0]
    # **** Predict final model from unseen dataset **** #
    predictions = predict_model(amp_model_final, data=data_unseen)
    # **** Accuracy Score **** #
    new_model_score = check_metric(
        predictions.amp_motor, predictions.Label, "R2")

    # **** Load log models **** #
    log = pd.read_excel("pl_models/model_log.xlsx")
    index = log[log.Model_Name == MODEL].index

    if new_model_score > exist_model_score:
        # **** Save Model **** #
        save_model(amp_model_final, MODEL, verbose=False)
        # **** Assing model algorithm name **** #
        log.loc[index, "Algorithm"] = algorithm_name
        # **** Assing model accuracy score **** #
        log.loc[index, "Accuracy"] = new_model_score
        # **** Assing model create date **** #
        log.loc[index, "Version"] = pd.Timestamp.today().strftime("%Y-%m-%d")
        # **** Set accuracy score **** #
        accuracy_score = new_model_score
    else:
        log.loc[index, "Accuracy"] = exist_model_score
    log.to_excel("pl_models/model_log.xlsx", index=False)

    return algorithm_name, accuracy_score


def create_models():
    # **** Assign Line **** #
    pl_line = [3]
    # pl_line = range(1,6)
    logs = []
    for pl in pl_line:
        # **** Get data PL **** #
        df = pd.read_excel(f"pl_cleaned/cleaned_pl{pl}.xlsx")

        # **** Machine Learning Process **** #
        st_alg, st_score = steam_ratio_model(pl, df, 0.9)
        amp_alg, amp_score = amp_motor_model(pl, df, 0.9)

        # **** Log Status **** #
        text_log = f'''
        ------------------------------
        PL {pl}
        ------------------------------
        Model Steam Ratio
        Algorithm : {st_alg}
        Accuracy Score : {st_score}
        ------------------------------
        Model Amp Motor
        Algorithm : {amp_alg}
        Accuracy Score : {amp_score}'''

        logs.append(text_log)

    [print(log) for log in logs]

    return


def prediction(pl):
    print(f'Prediction for PL{pl}')
    # **** API Retrieve Job **** #
    url = f'http://10.4.10.248:29201/apiPellet/WebService1.asmx/Call_Pellet_job?PL_Number={pl}'
    response = requests.get(url)
    # **** Extract XML **** #
    root = ET.fromstring(response.text)
    for tag in root.iter('Table1'):
        job_api = {}
        for t in tag:
            job_api[t.tag] = t.text

    # **** Create Job detail **** #
    job = pd.DataFrame([job_api])






    # CASE: JOB ACTIVE
    if job.loc[0, 'formular_date'] != '0':

        # **** Format formular date **** #
        job.formular_date = [datetime.strptime(t, "%d/%m/%y") for t in job.formular_date]
        # **** Get formular month **** #
        job["formular_month"] = [i.month for i in job.formular_date]

        # **** Get parameter to search form master **** #
        job_code = job.formular_code[0]
        job_fpqf = job.fpqf[0]
        job_month = [job.formular_month[0]] + [job.formular_month[0] + 1]

        # **** Get master data **** #
        df = pd.read_excel(f'pl_cleaned/cleaned_pl{job.loc[0,"pl_name"]}.xlsx')
        df.formular_code = df.formular_code.astype(str)
        # **** Filter by parameter **** #
        exist_code = df[(df.formular_code == job_code) & (df.formular_month.isin(job_month))].copy()
      
        # CASE: EXIST SKU
        if len(exist_code):
            data = exist_code

        else:
            # **** Find formular group of job code **** #
            group = pd.read_excel("pl_cleaned/FeedGroup.xlsx")
            job_group = group[group.formular_code == job_code]['formular_group'].values[0]
            # CASE: EXIST SKU IN FORMULAR GROUP
            if len(job_group):
                # **** Get fpqf from master **** #
                fpqf_list = list(set(df.fpqf))
                # **** Find nearest fpqf of job and master **** #
                near_list = [abs(f-job_fpqf) for f in fpqf_list]
                fpqf = fpqf_list[near_list.index(min(near_list))]
                # **** Filter by group and fpqf **** #
                data = df[(df.formular_group == job_group) & (df.fpqf ==fpqf )& (df.formular_month.isin(job_month))].copy()
            # CASE: NEW SKU IN FORMULAR GROUP
            else:
                data = df
                
        




        # **** Create parameter to return API **** #
        parameter = {}
        # **** Get ton per hr usually and max **** #
        ton_mean = data.ton_per_hr.mean()
        ton_mode = data.ton_per_hr.mode()[0]
        ton_max = data.ton_per_hr.max()

        # **** Assign to parameter **** #
        parameter["ton_per_hr_use"] = round(max([ton_mean, ton_mode]), 1)
        parameter["ton_per_hr_max"] = round(ton_max, 1)

        # **** Create dataset to predict **** #
        plc = job[["pl_name", "formular_code", "fpqf", "density", "fat_target_percent",
                    "molass_target_percent", "bar_steam", "formular_month", ]].copy()
        # **** Get formular group **** #
        plc["formular_group"] = data.formular_group.mode()[0]
        # **** 2 case = usually / max **** #
        plc_use = plc.copy()
        plc_max = plc.copy()

        # **** Import steam Model & amp Model **** #
        steam_model = load_model(f"Model_PL{pl}_Steam_Ratio", verbose=False)
        amp_model = load_model(f"Model_PL{pl}_Amp_Motor", verbose=False)

        # **** Predict usually case **** #
        plc_use["ton_per_hr"] = parameter["ton_per_hr_use"]
        result = predict_model(steam_model, data=plc_use)
        plc_use["steam_kg_ton"] = result.Label[0].round(1)
        result = predict_model(amp_model, data=plc_use)
        plc_use["amp_motor"] = int(result.Label[0])

        # **** Predict max case **** #
        plc_max["ton_per_hr"] = parameter["ton_per_hr_max"]
        result = predict_model(steam_model, data=plc_max)
        plc_max["steam_kg_ton"] = result.Label[0].round(1)
        result = predict_model(amp_model, data=plc_max)
        plc_max["amp_motor"] = int(result.Label[0])

        # **** Parameter Tuning **** #
        const = df[(df["operate_mode"] == 1) &
                    (df.formular_code == job_code)]

        # CASE: CALCULATE K VALUE OF STEAM RATIO           
        if len(const):
            # **** K value of steam ration **** #
            k_steam = round((const.steam_kg_ton_target -
                            const.steam_kg_ton).mean(), 1)
            plc_use["steam_kg_ton"] = plc_use["steam_kg_ton"] + k_steam
            plc_max["steam_kg_ton"] = plc_max["steam_kg_ton"] + k_steam

        # **** Set safety factor of max amp motor is 0.9 **** #
        plc_use["amp_motor"] = int(plc_use["amp_motor"] / 0.9)
        plc_max["amp_motor"] = int(plc_max["amp_motor"] / 0.9)
        # **** Assign to parameter **** #
        parameter["steam_kg_ton_use"] = plc_use.loc[0,
                                                    "steam_kg_ton"].round(1)
        parameter["steam_kg_ton_max"] = plc_max.loc[0,
                                                    "steam_kg_ton"].round(1)
        parameter["max_amp_motor_use"] = plc_use.loc[0, "amp_motor"]
        parameter["max_amp_motor_max"] = plc_max.loc[0, "amp_motor"]

        # **** API to return prediction **** #
        url = f"http://10.4.10.248:29201/apiPellet/WebService1.asmx/Pellet_predicted?pl_number={pl}&job_id={job.job_id[0]}&ton_hr={parameter['ton_per_hr_use']}&steam_ratio={parameter['steam_kg_ton_use']}&AMP={parameter['max_amp_motor_use']}&ton_hr_tryhard={parameter['ton_per_hr_max']}&steam_ratio_tryhard={parameter['steam_kg_ton_max']}&AMP_tryhard={parameter['max_amp_motor_max']}&state=BEST"

        print('-'*20)
        print(f'Job ID : {job.job_id[0]}')
        print(f'Formular Code : {job.formular_code[0]}')
        print(f'Formular Name : {job.formular_name[0]}')
        print('-'*20)
        print('Usually Case')
        print(f"ton_per_hr : {parameter['ton_per_hr_use']}")
        print(f"steam_ratio : {parameter['steam_kg_ton_use']}")
        print(f"max_amp_motor : {parameter['max_amp_motor_use']}")
        print('-'*20)
        print('Maximum Case')
        print(f"ton_per_hr : {parameter['ton_per_hr_max']}")
        print(f"steam_ratio : {parameter['steam_kg_ton_max']}")
        print(f"max_amp_motor : {parameter['max_amp_motor_max']}")
        print('-'*20)

        response = requests.get(url)

        # CASE: SEND VALUE SUCCESS
        if response.status_code == 200:
            print(f'Prediction for PL{pl} success')
        # CASE: SEND VALUE FAILED
        else:
            print('System Alarm : Return to API failed')

    # CASE: JOB DEACTIVE
    else:
        print(f'System Alarm : PL{pl} no job')
    return


# retrieve_data()
# ETL()
# create_models()
prediction(3)
