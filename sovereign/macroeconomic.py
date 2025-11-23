# Script with functions for macroeconomic modelling
from openpyxl import load_workbook
import pandas as pd
import subprocess
import datetime
from pathlib import Path
import os

def update_calibration_parameters(sheet, parameter, new_value):
    '''
    This function updates the calibration sheet in DIGNAD.
    '''

    for row in sheet.iter_rows():
        for cell in row:
            if cell.value == parameter:
                # Assuming the value needs to be updated in the cell right after the parameter
                target_cell = sheet.cell(row=cell.row, column=cell.column + 1)
                try:
                    # Convert to float first
                    target_cell.value = float(new_value)
                except ValueError:
                    # if above doesn't work, save as string
                    target_cell.value = new_value
                return True  # Return after the first match to avoid unnecessary updates
    return False  # Return False if parameter not found

def update_natural_hazard_parameters(sim_start_year, nat_disaster_year, recovery_period, tradable_impact,
                                    nontradable_impact, reconstruction_efficiency, public_debt_premium,
                                    public_impact, private_impact, share_tradable):
    '''
    This function returns a dictionary that will be used to populate the Disasters sheet in DIGNAD.
    It takes as input the 9 parameters the user typically has to set in the Disasters sheet.
    '''

    excel_updates = {
    (3, 4): nat_disaster_year - sim_start_year, # C4 cell update
    (4, 4): nat_disaster_year, # D4 cell update
    (4, 7): tradable_impact, # D7 cell update
    (4, 8): nontradable_impact, # D8 cell update
    (4, 9): reconstruction_efficiency, # D9 cell update
    (4, 10): public_debt_premium, # D10 cell update
    (4, 11): public_impact, # D11 cell update
    (4, 12): private_impact, # D12 cell update
    (4, 13): share_tradable, # D13 cell update
    (3, 17): nat_disaster_year, # C17 cell update
    (4, 17): nat_disaster_year, # D17 cell update
    (3, 18): nat_disaster_year + recovery_period, # C18 cell update
    (4, 18): nat_disaster_year + recovery_period, # D18 cell update
    (3, 20): nat_disaster_year, # C20 cell update
    (3, 21): nat_disaster_year + recovery_period, # C21 cell update
    (3, 23): nat_disaster_year, # C23 cell update
    (3, 24): nat_disaster_year + recovery_period, # C24 cell update
    (3, 26): nat_disaster_year, # C26 cell update
    (3, 27): nat_disaster_year + recovery_period # C27 cell update
    }

    return excel_updates

def prepare_DIGNAD(calibration_csv, adaptation_cost):
    '''
    This function prepares DIGNAD for being run in a simulation.
    It updates the calibration sheet with parameters from a CSV file.
    It also incorporates adaptation costs as an exogenous cost in DIGNAD. 
    Here adaptation cost is assumed to be a fixed annual investment over 5 
    years (before the disaster strikes). This is incorporated as a % of GDP.
    '''

    # Calculate adaptation cost as % of GDP
    Thai_GDP = 495.65 # in billion USD, 2022 value
    # Calculate annual adaptation cost as % of GDP (over 5 years)
    annual_adaptation_cost = (adaptation_cost / 5) / Thai_GDP * 100

    ### 1. Load the original Excel file - this is where all DIGNAD parameters are set
    # Find parent directory
    root = Path.cwd().parent
    DIGNAD_root = os.path.join(root, "DIGNAD", "DIGNAD_Toolkit", "DIGNAD_Toolkit")
    excel_file = os.path.join(DIGNAD_root, "input_DIG-ND.xlsx") # Assuming DIGNAD folder is in correct location
    wb = load_workbook(excel_file)

    ### 2. Load the CSV with calibration parameters
    calibration_df = pd.read_csv(calibration_csv)

    ### 3. Set calibration parameters
    sheet = wb['Calibration']
    # Iterate over the calibration DataFrame rows
    for index, row in calibration_df.iterrows():
        parameter = row['Parameters']  # The column name in your CSV for the parameter names
        new_value = row['Values']       # The column name in your CSV for the new values
        updated = update_calibration_parameters(sheet, parameter, new_value)
        if not updated:
            print(f"Parameter '{parameter}' not found in the Excel sheet.")

    ### 4. Incorporate adaptation costs
    adaptation_cost_dic = {
        (5, 4): annual_adaptation_cost,  # E4 cell update
        (5, 5): annual_adaptation_cost,  # E5 cell update
        (5, 6): annual_adaptation_cost,  # E6 cell update
        (5, 7): annual_adaptation_cost,  # E7 cell update
        (5, 8): annual_adaptation_cost   # E8 cell update
    }
    sheet = wb['Exogenous_series']
    for (col, row), value in adaptation_cost_dic.items():
        cell = sheet.cell(row=row, column=col)
        cell.value = value

    wb.save(excel_file)

def run_DIGNAD(sim_start_year, nat_disaster_year, recovery_period, tradable_impact, nontradable_impact,
                reconstruction_efficiency, public_debt_premium, public_impact, private_impact, share_tradable):
    '''
    This function runs on instance of DIGNAD with a prespecified calibration csv.
    Parameters passed to the function are the natural hazard parameters.
    Function outputs a list of GDP losses - from 2021 - 2040
    Need to set filepaths for own DIGNAD installation.
    '''

    ### 1. Set DIGNAD directory path (load Excel file)
    root = Path.cwd().parent
    DIGNAD_root = os.path.join(root, "DIGNAD", "DIGNAD_Toolkit", "DIGNAD_Toolkit")
    excel_file = os.path.join(DIGNAD_root, "input_DIG-ND.xlsx") # Assuming DIGNAD folder is in correct location
    wb = load_workbook(excel_file)

    ### 2. Update disasters sheet
    natdisaster_params = update_natural_hazard_parameters(sim_start_year, nat_disaster_year, recovery_period, tradable_impact,
                                                                nontradable_impact, reconstruction_efficiency,
                                                                public_debt_premium, public_impact, private_impact, share_tradable)
    sheet = wb['Disasters']
    for (col, row), value in natdisaster_params.items():
        cell = sheet.cell(row=row, column=col)
        cell.value = value

    ### 3. Save Excel Workbook
    wb.save(excel_file)

    ### 4. Run Matlab (note: Matlab should be added to PATH)
    matlab_script = os.path.join(DIGNAD_root, "simulate.m")
    result = subprocess.call(["matlab", "-batch", "run('" + matlab_script + "')"])
    if int(result) != 0:
        print("MATLAB script not executed succesfully")
        return None, None

    ### 5. Read results from Excel sheet
    # Get today's date as that is the name of file and directory
    today = datetime.datetime.today().strftime("%d%b%Y")
    file_path = os.path.join(DIGNAD_root, "Excel output", f"{today}", f"Model_output_{today}.xlsx")
    df = pd.read_excel(file_path)
    years = list(df.iloc[:, 0])
    gdp_impact = list(df.iloc[:, 1])

    return gdp_impact, years