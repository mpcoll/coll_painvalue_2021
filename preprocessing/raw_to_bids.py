from nipype.interfaces.dcm2nii import Dcm2niix
import os
from os.path import join as osj
import json
import glob
import pandas as pd
import convert_eprime as ceprime
import numpy as np

#  ________________________________________________________________________
# Parameters


class parameters:
    def __init__(self):
        self.datadrive = "/data"  # Change this
        self.rawpath = osj(self.datadrive, 'raw')
        self.bidspath = osj(self.datadrive, 'source')
        self.behavpath = osj(self.datadrive, 'raweprime')
        self.outpath = osj(self.datadrive, 'derivatives')
        self.workpath = osj(self.datadrive,
                            '/data/derivatives/scratch')
        self.inout = {'anat': ['03-acq-mgh-1mm-4e_T1w RMS/echo_1.87'],
                      'func': ['04-task-p2-s3-3mm_Run-1',
                               '06-task-p2-s3-3mm_Run-2',
                               '08-task-p2-s3-3mm_Run-3',
                               '10-task-p2-s3-3mm_Run-4',
                               '12-task-p2-s3-3mm_Run-5'],
                      'fmap': ['14-task-gre_field_mapping/echo_4.92',
                               '14-task-gre_field_mapping/echo_7.38',
                               '15-task-gre_field_mapping/echo_7.38']
                      }

        for p in [self.bidspath, self.outpath, self.workpath]:
            if not os.path.exists(p):
                os.mkdir(p)

        # Get all subs to process
        self.subs = [dI for dI in os.listdir(
            self.rawpath) if os.path.isdir(os.path.join(self.rawpath, dI))]



p = parameters()
p.subs = [s for s in p.subs if 'NPNG' in s]
#  ________________________________________________________________________
# Make dataset level JSON description files


# Function to write to json files
def writetojson(outfile, path, content):
    data = os.path.join(path, outfile)
    if not os.path.isfile(data):
        with open(data, 'w') as outfile:
            json.dump(content, outfile)
    else:
        print('File ' + outfile + ' already exists.')


# Create description files
dataset_description = {"Name": "NoPainNoGain McGill - 2019",
                       "BIDSVersion": "1.2.0",
                       "Authors": ["H Slimani", "M Roy", "MP Coll"]}

t1_description = {
                  "Modality": "MR",
                  "MagneticFieldStrength": 3,
                  "ImagingFrequency": 123.26,
                  "Manufacturer": "Siemens",
                  "ManufacturersModelName": "Prisma_fit",
                  "InstitutionName": "IUGM",
                  "InstitutionalDepartmentName": "Department",
                  "InstitutionAddress": "Chemin_Queen-Mary_4565_CA_H3W_1W5",
                  "DeviceSerialNumber": "167006",
                  "StationName": "MRC35049",
                  "BodyPartExamined": "BRAIN",
                  "PatientPosition": "HFS",
                  "ProcedureStepDescription": "fMRI_Rainville",
                  "SoftwareVersions": "syngo_MR_E11",
                  "MRAcquisitionType": "3D",
                  "SeriesDescription": "acq-mgh-1mm-4e_T1w",
                  "ProtocolName": "acq-mgh-1mm-4e_T1w",
                  "ScanningSequence": "GR_IR",
                  "SequenceVariant": "SK_SP_MP",
                  "ScanOptions": "IR",
                  "SequenceName": "tfl3d4_16ns",
                  "ImageType": ["ORIGINAL", "PRIMARY", "M", "ND", "NORM"],
                  "SeriesNumber": 2,
                  "AcquisitionTime": "15:53:55.505000",
                  "AcquisitionNumber": 1,
                  "SliceThickness": 1,
                  "SAR": 0.0493425,
                  "EchoNumber": 1,
                  "EchoTime": 0.00187,
                  "RepetitionTime": 2.2,
                  "InversionTime": 1.1,
                  "FlipAngle": 8,
                  "PartialFourier": 1,
                  "BaseResolution": 256,
                  "ShimSetting": [619,
                                  -7199,
                                  -9109,
                                  77,
                                  24,
                                  -19,
                                  -12,
                                  -20],
                  "TxRefAmp": 239.192,
                  "PhaseResolution": 1,
                  "ReceiveCoilName": "Head_32",
                  "ReceiveCoilActiveElements": "HEA;HEP",
                  "PulseSequenceDetails": "%CustomerSeq%_tfl_mgh_epinav_ABCD",
                  "WipMemBlock": "Prisma_epi_moco_navigator_ABCD_tfl.prot",
                  "RefLinesPE": 32,
                  "ConsistencyInfo": "N4_VE11C_LATEST_20160120",
                  "PercentPhaseFOV": 100,
                  "PhaseEncodingSteps": 255,
                  "AcquisitionMatrixPE": 256,
                  "ReconMatrixPE": 256,
                  "ParallelReductionFactorInPlane": 2,
                  "PixelBandwidth": 500,
                  "DwellTime": 3.9e-06,
                  "ImageOrientationPatientDICOM": [0,
                                                   1,
                                                   0,
                                                   0,
                                                   0,
                                                   -1],
                  "InPlanePhaseEncodingDirectionDICOM": "ROW"}

# From DICOM headers
task_description = {"Modality": "MR",
                    "MagneticFieldStrength": 3,
                    "ImagingFrequency": 123.259,
                    "Manufacturer": "Siemens",
                    "ManufacturersModelName": "Prisma_fit",
                    "InstitutionName": "IUGM",
                    "InstitutionalDepartmentName": "Department",
                    "InstitutionAddress": "Montreal_District_CA_H3W_1W5",
                    "DeviceSerialNumber": "167006",
                    "StationName": "MRC35049",
                    "BodyPartExamined": "BRAIN",
                    "PatientPosition": "HFS",
                    "ProcedureStepDescription": "fMRI_Rainville",
                    "SoftwareVersions": "syngo_MR_E11",
                    "MRAcquisitionType": "2D",
                    "SeriesDescription": "task-p2-s3-3mm_Run-1",
                    "ProtocolName": "task-p2-s3-3mm_Run-1",
                    "ScanningSequence": "EP",
                    "SequenceVariant": "SK",
                    "ScanOptions": "FS",
                    "SequenceName": "_epfid2d1_64",
                    "ImageType": ["ORIGINAL", "PRIMARY", "M", "ND", "MOSAIC"],
                    "SeriesNumber": 4,
                    "AcquisitionTime": "16:19:39.887500",
                    "AcquisitionNumber": 1,
                    "SliceThickness": 3,
                    "SpacingBetweenSlices": 3,
                    "SAR": 0.0806607,
                    "EchoTime": 0.02,
                    "RepetitionTime": 0.832,
                    "FlipAngle": 58,
                    "PartialFourier": 1,
                    "BaseResolution": 64,
                    "ShimSetting": [
                                    651,
                                    -7202,
                                    -8848,
                                    523,
                                    129,
                                    43,
                                    74,
                                    -42	],
                    "TxRefAmp": 237.518,
                    "PhaseResolution": 1,
                    "ReceiveCoilName": "Head_32",
                    "ReceiveCoilActiveElements": "HEA;HEP",
                    "PulseSequenceDetails": "%SiemensSeq%_ep2d_bold",
                    "ConsistencyInfo": "N4_VE11C_LATEST_20160120",
                    "MultibandAccelerationFactor": 3,
                    "PercentPhaseFOV": 100,
                    "EchoTrainLength": 31,
                    "PhaseEncodingSteps": 63,
                    "AcquisitionMatrixPE": 64,
                    "ReconMatrixPE": 64,
                    "BandwidthPerPixelPhaseEncode": 45.956,
                    "ParallelReductionFactorInPlane": 2,
                    "EffectiveEchoSpacing": 0.000339999,
                    "DerivedVendorReportedEchoSpacing": 0.000679998,
                    "TotalReadoutTime": 0.0214199,
                    "PixelBandwidth": 1735,
                    "DwellTime": 4.5e-06,
                    "PhaseEncodingDirection": "j-",
                    "SliceTiming": [
                                    0,
                                    0.43,
                                    0.0475,
                                    0.4775,
                                    0.095,
                                    0.525,
                                    0.1425,
                                    0.5725,
                                    0.19,
                                    0.62,
                                    0.2375,
                                    0.6675,
                                    0.285,
                                    0.715,
                                    0.3325,
                                    0.7625,
                                    0.38,
                                    0,
                                    0.43,
                                    0.0475,
                                    0.4775,
                                    0.095,
                                    0.525,
                                    0.1425,
                                    0.5725,
                                    0.19,
                                    0.62,
                                    0.2375,
                                    0.6675,
                                    0.285,
                                    0.715,
                                    0.3325,
                                    0.7625,
                                    0.38,
                                    0,
                                    0.43,
                                    0.0475,
                                    0.4775,
                                    0.095,
                                    0.525,
                                    0.1425,
                                    0.5725,
                                    0.19,
                                    0.62,
                                    0.2375,
                                    0.6675,
                                    0.285,
                                    0.715,
                                    0.3325,
                                    0.7625,
                                    0.38	],
                    "ImageOrientationPatientDICOM": [
                                                    1,
                                                    0,
                                                    0,
                                                    0,
                                                    0.920505,
                                                    -0.390731],
                    "InPlanePhaseEncodingDirectionDICOM": "COL",
                    "ConversionSoftware": "dcm2niix",
                    "ConversionSoftwareVersion": "v1.0.20181125",
                    'TaskName': 'npng'
                    }


# Write dataset description
writetojson('T1w.json', p.bidspath, t1_description)
writetojson('dataset_description.json', p.bidspath, dataset_description)
writetojson('task-npng_bold.json', p.bidspath, task_description)

# %% ________________________________________________________________________
# First pass, convert 3d nii to 4d nii.gz

converter = Dcm2niix()
for s in p.subs:
    outsub = osj(p.bidspath, 'sub-' + s[-2:len(s)])
    source = osj(p.rawpath, s)
    if not os.path.exists(outsub):
        os.mkdir(outsub)
        for bidsdir, subdir in p.inout.items():
            os.mkdir(osj(outsub, bidsdir))
            for sd in subdir:
                # Remove stupid space in some filenames
                if os.path.exists(osj(source, '03-acq-mgh-1mm-4e_T1w RMS')):
                    os.rename(osj(source, '03-acq-mgh-1mm-4e_T1w RMS'),
                              osj(source, '03-acq-mgh-1mm-4e_T1wRMS'))
                # Correct some specific deviations
                converter.inputs.source_dir = osj(source, sd).replace(' ', '')
                converter.inputs.compress = 'y'
                converter.inputs.output_dir = osj(outsub, bidsdir)
                converter.cmdline
                converter.inputs.bids_format = False
                os.system(converter.cmdline)

    else:
        print(outsub + ' already exists, dicom conversion skipped')

print('DICOM -> NIFTI CONVERSION DONE')

#  ________________________________________________________________________
# Second pass: Rename and sort .nii.gz

# T1 Files
for s in p.subs:
    bidssubfold = osj(p.bidspath, 'sub-' + s[-2:len(s)])
    bidssub = 'sub-' + s[-2:len(s)]

    # Anat
    t1files = glob.glob(osj(bidssubfold, 'anat', '*T1*.nii.gz'),
                        recursive=False)
    os.rename(t1files[0], osj(bidssubfold, 'anat', bidssub + '_T1w.nii.gz'))

    # Field maps
    mag1 = glob.glob(osj(bidssubfold, 'fmap', '*echo_4*gre_field*.nii.gz'),
                     recursive=False)
    if len(mag1) > 0:

        os.rename(mag1[0], osj(bidssubfold, 'fmap',
                               bidssub + '_magnitude1.nii.gz'))

    mag2 = glob.glob(osj(bidssubfold, 'fmap', 'echo_7*gre_field*.nii.gz'),
                     recursive=False)

    if len(mag2) > 0:

        os.rename(mag2[0], osj(bidssubfold, 'fmap',
                               bidssub + '_magnitude2.nii.gz'))

    phasediff = glob.glob(osj(bidssubfold, 'fmap', '*gre_field*_ph.nii.gz'),
                          recursive=False)

    if len(phasediff) > 0:

        os.rename(phasediff[0], osj(bidssubfold, 'fmap',
                                    bidssub + '_phasediff.nii.gz'))

    # BOLD
    for runs in range(1, 6):
        bold = glob.glob(osj(bidssubfold, 'func',
                             '*Run-' + str(runs) + '*.nii.gz'),
                         recursive=False)

        if len(bold) > 0:

            os.rename(bold[0], osj(bidssubfold, 'func',
                                   bidssub + '_task-npng_run-'
                                   + str(runs) + '_bold.nii.gz'))

    # Make JSON file
    fmap = {
        "Modality": "MR",
        "MagneticFieldStrength": 3,
        "ImagingFrequency": 123.259,
        "Manufacturer": "Siemens",
        "ManufacturersModelName": "Prisma_fit",
        "InstitutionName": "IUGM",
        "InstitutionalDepartmentName": "Department",
        "InstitutionAddress": "Montreal_District_CA_H3W_1W5",
        "DeviceSerialNumber": "167006",
        "StationName": "MRC35049",
        "BodyPartExamined": "BRAIN",
        "PatientPosition": "HFS",
        "ProcedureStepDescription": "fMRI_Rainville",
        "SoftwareVersions": "syngo_MR_E11",
        "MRAcquisitionType": "2D",
        "SeriesDescription": "task-gre_field_mapping",
        "ProtocolName": "task-gre_field_mapping",
        "ScanningSequence": "GR",
        "SequenceVariant": "SP",
        "SequenceName": "_fm2d2r",
        "ImageType": ["ORIGINAL", "PRIMARY", "M", "ND"],
        "SeriesNumber": 14,
        "AcquisitionTime": "17:01:16.015000",
        "AcquisitionNumber": 1,
        "SliceThickness": 3,
        "SpacingBetweenSlices": 3,
        "SAR": 0.139589,
        "RepetitionTime": 0.54,
        "FlipAngle": 60,
        "PartialFourier": 1,
        "BaseResolution": 64,
        "ShimSetting": [
                        651,
                        -7202,
                        -8848,
                        523,
                        129,
                        43,
                        74,
                        -42	],
        "TxRefAmp": 237.518,
        "PhaseResolution": 1,
        "ReceiveCoilName": "Head_32",
        "ReceiveCoilActiveElements": "HEA;HEP",
        "PulseSequenceDetails": "%SiemensSeq%_gre_field_mapping",
        "ConsistencyInfo": "N4_VE11C_LATEST_20160120",
        "PercentPhaseFOV": 100,
        "PhaseEncodingSteps": 64,
        "AcquisitionMatrixPE": 64,
        "ReconMatrixPE": 64,
        "PixelBandwidth": 290,
        "DwellTime": 2.69e-05,
        "PhaseEncodingDirection": "j-",
        "ImageOrientationPatientDICOM": [
                                            1,
                                            0,
                                            0,
                                            0,
                                            0.920505,
                                            -0.390731	],
        "InPlanePhaseEncodingDirectionDICOM": "COL",
        "ConversionSoftware": "dcm2niix",
        "EchoTime2": 0.00738,
        "EchoTime1": 0.00492,  # From DICOM tag 0018, 0081
        'IntendedFor':  ["func/" + bidssub
                         + "_task-npng_run-1_bold.nii.gz",
                         "func/" + bidssub
                         + "_task-npng_run-2_bold.nii.gz",
                         "func/" + bidssub
                         + "_task-npng_run-3_bold.nii.gz",
                         "func/" + bidssub
                         + "_task-npng_run-4_bold.nii.gz",
                         "func/" + bidssub
                         + "_task-npng_run-5_bold.nii.gz"]}

    writetojson(osj(bidssubfold, 'fmap', bidssub + '_phasediff.json'),
                bidssubfold,  fmap)


#  ________________________________________________________________________
# Behavioural data

def ms_to_onsets(list, scan_start, dur=0):
    if dur == 0:
        out = [round(x/1000 - scan_start, 5) for x in list]
    else:
        out = [round(x/1000, 5) for x in list]

    return out


for s in p.subs:
    bidssubfold = osj(p.bidspath, 'sub-' + s[-2:len(s)])
    bidssub = 'sub-' + s[-2:len(s)]
    sourcedir = osj(p.rawpath, s)
    behavfiles = glob.glob(osj(sourcedir, '*.txt'), recursive=False)
    runs = []
    for idx, b in enumerate(behavfiles):
        # Create new data frame
        events = pd.DataFrame()

        # Find out which run
        run = b.split(os.sep)[-1][16]

        # Convert eprime to csv
        ceprime.text_to_csv(b, b.replace('.txt', '.csv'))

        # Load
        edat = pd.read_csv(b.replace('.txt', '.csv'))

        # Trials info
        events['pain_level'] = edat.stimulus
        events['pain_rank'] = edat.multiplyer
        events['money_level'] = edat.comp
        events['money_rank'] = round(edat.comp/1.11) + 1
        events['intensity_level'] = edat.mA
        events['painfirst'] = [0 if proc == 'auction2' else 1
                               for proc in edat.Procedure]
        events = events.dropna()

        # Onsets
        scan_start = edat['instructions.RTTime'][0]/1000  # Get trigger

        # Offer specific values
        choice, choice_ons, feed_ons, offer2_ons, offer2_dur = [], [], [], [], []
        for t in range(len(edat)-1):
            if str(edat['choice1.ACC'][t]) == 'nan':
                choice.append(edat['choice2.ACC'][t])
                choice_ons.append(edat['choice2.RTTime'][t])
                feed_ons.append(edat['feedback2.OnsetTime'][t])
                offer2_ons.append(edat['choice2.OnsetTime'][t])
                offer2_dur.append(edat['choice2.RT'][t])
            else:
                choice.append(edat['choice1.ACC'][t])
                feed_ons.append(edat['feedback1.OnsetTime'][t])
                offer2_ons.append(edat['choice1.OnsetTime'][t])
                offer2_dur.append(edat['choice1.RT'][t])
                choice_ons.append(edat['choice1.RTTime'][t])

        events.loc[:, 'accept'] = choice
        events.loc[:, 'feedback_onset'] = ms_to_onsets(feed_ons, scan_start)
        events.loc[:, 'choice_onset'] = ms_to_onsets(choice_ons, scan_start)
        events.loc[:, 'offer2_onset'] = ms_to_onsets(offer2_ons, scan_start)
        events.loc[:, 'offer2_duration'] = ms_to_onsets(offer2_dur,
                                                        scan_start, dur=1)

        # Offer 1
        offer1_onset = (list(edat['intensity.OnsetTime'].dropna())
                        + list(edat['compensation.OnsetTime'].dropna()))
        offer1_onset.sort()
        events.loc[:, 'offer1_onset'] = ms_to_onsets(offer1_onset, scan_start)

        # Fixations onsets and durations
        delay1_ons, delay1_dur, delay2_ons, delay2_dur = [], [], [], []
        delay3_ons, delay3_dur = [], []
        for t in range(len(edat)-1):
            delay1_dur.append(edat.tid1[t])
            delay2_dur.append(edat.tid2[t])
            delay3_dur.append(edat.tid3[t])
            delay1_ons.append(edat['fixation1' + str(int(edat.tid1[t]))
                                   + '.OnsetTime'][t])
            delay2_ons.append(edat['fixation2' + str(int(edat.tid2[t]))
                                   + '.OnsetTime'][t])
            delay3_ons.append(edat['fixation3' + str(int(edat.tid3[t]))
                                   + '.OnsetTime'][t])

        events.loc[:, 'delay1_onset'] = ms_to_onsets(delay1_ons, scan_start)
        events.loc[:, 'delay2_onset'] = ms_to_onsets(delay2_ons, scan_start)
        events.loc[:, 'delay3_onset'] = ms_to_onsets(delay3_ons, scan_start)
        events.loc[:, 'delay1_duration'] = delay1_dur
        events.loc[:, 'delay2_duration'] = delay2_dur
        events.loc[:, 'delay3_duration'] = delay3_dur

        # Shock onset
        shock_onset = list(edat['fixation00.OnsetTime'].dropna())
        events.loc[:, 'shock_onset'] = ms_to_onsets(shock_onset, scan_start)

        # Shock onset
        shock_duration = list(edat['temps'].dropna())
        events.loc[:, 'shock_duration'] = ms_to_onsets(shock_duration,
                                                       scan_start, dur=1)

        # Add pain/money received modulator
        # if refused pain == 0 otherwise pain_level
        pain_received = np.where(events['accept'] == 0, 0, events['pain_rank'])
        # if accepted == money level otherwise 0
        money_received = np.where(events['accept'] == 1, events['money_level'],
                                  0)
        pain_refused = np.where(events['accept'] == 0, events['pain_rank'],
                                0)
        money_refused = np.where(events['accept'] == 0, events['money_level'],
                                 0)

        events['pain_received'] = pain_received
        events['money_received'] = money_received
        events['pain_refused'] = pain_refused
        events['money_refused'] = money_refused

        # Make mock onsets and duration to satisfy bids standard that
        # usually take data in long format
        events['onset'] = 99
        events['duration'] = 99

        cols = list(events.columns.values)
        newcols = ['onset',
                   'duration',
                   'pain_level',
                   'pain_rank',
                   'money_level',
                   'pain_received',
                   'money_received',
                   'pain_refused',
                   'money_refused',
                   'intensity_level',
                   'painfirst',
                   'accept',
                   'feedback_onset',
                   'choice_onset',
                   'offer2_onset',
                   'offer2_duration',
                   'offer1_onset',
                   'delay1_onset',
                   'delay2_onset',
                   'delay3_onset',
                   'delay1_duration',
                   'delay2_duration',
                   'delay3_duration',
                   'shock_onset',
                   'shock_duration']
        # Reorder columns
        events = events[newcols]

        # Get ranked values for pain stimulation
        pval = list(set(events.pain_level))
        pval.sort()
        painrank, painrank5 = [], []
        for pl in events.pain_level:
            painrank.append((pval.index(pl)+1))
            if pval.index(pl)+1 in [1, 2]:
                painrank5.append(1)
            if pval.index(pl)+1 in [3, 4]:
                painrank5.append(2)
            if pval.index(pl)+1 in [5, 6]:
                painrank5.append(3)
            if pval.index(pl)+1 in [7, 8]:
                painrank5.append(4)
            if pval.index(pl)+1 in [9, 10]:
                painrank5.append(5)
        events.loc[:, 'pain_rank10l'] = painrank
        events.loc[:, 'pain_rank5l'] = painrank5

        # Get ranked values for pain stimulation 4 levels
        pval = list(set(events.pain_level))
        pval.sort()
        painrank, painrank4 = [], []
        for pl in events.pain_level:
            painrank.append((pval.index(pl)+1))
            if pval.index(pl)+1 in [1, 2]:
                painrank4.append(1)
            if pval.index(pl)+1 in [3, 4]:
                painrank4.append(2)
            if pval.index(pl)+1 in [5, 6, 7]:
                painrank4.append(3)
            if pval.index(pl)+1 in [8, 9, 10]:
                painrank4.append(4)
            if pval.index(pl)+1 in []:
                painrank4.append(5)
        events.loc[:, 'pain_rank4l'] = painrank4

        # Get ranked values for money offer
        mval = list(set(events.money_level))
        mval.sort()
        money_rank, money_rank5 = [], []
        for ml in events.money_level:
            money_rank.append((mval.index(ml)+1))
            if mval.index(ml)+1 in [1, 2]:
                money_rank5.append(1)
            if mval.index(ml)+1 in [3, 4]:
                money_rank5.append(2)
            if mval.index(ml)+1 in [5, 6]:
                money_rank5.append(3)
            if mval.index(ml)+1 in [7, 8]:
                money_rank5.append(4)
            if mval.index(ml)+1 in [9, 10]:
                money_rank5.append(5)
        events.loc[:, 'money_rank10l'] = money_rank
        events.loc[:, 'money_rank5l'] = money_rank5

        # Make some additional columns depending on the trials
        # Split all columns according to choice and first offer
        oldcols = list(events.columns.values)
        for col in oldcols:
            events[col + '_painfirst'] = np.where(events.painfirst == 1,
                                                  events[col],
                                                  9999)
            events[col + '_moneyfirst'] = np.where(events.painfirst == 0,
                                                   events[col],
                                                   9999)
            events[col + '_accept'] = np.where(events.accept == 1,
                                               events[col],
                                               9999)
            events[col + '_reject'] = np.where(events.accept == 0,
                                               events[col],
                                               9999)

        # Add pain and money levels for anticipation and shock (5)
        for lev, val in enumerate([1, 2, 3, 4, 5]):
            pon = np.where((events.painfirst == 1)
                           & ((events.pain_rank5l == val)
                              | (events.pain_rank5l == val)),
                           events['offer1_onset'],
                           9999)
            mon = np.where((events.painfirst == 0)
                           & ((events.money_rank5l == val)
                              | (events.money_rank5l == val)),
                           events['offer1_onset'],
                           9999)

            shck = np.where(events.pain_rank5l == val,
                            events['shock_onset_accept'],
                            9999)
            d3ons = np.where(events.pain_rank5l == val,
                             events['delay3_onset_accept'],
                             9999)
            d3dur = np.where(events.pain_rank5l == val,
                             events['delay3_duration_accept'],
                             9999)

            events['o1_ons_money_5l_' + str(lev + 1)] = mon
            events['o1_ons_pain_5l_' + str(lev + 1)] = pon
            events['shock_ons_accept_5l_' + str(lev + 1)] = shck
            events['delay3_ons_accept_5l_' + str(lev + 1)] = d3ons
            events['delay3_dur_accept_5l_' + str(lev + 1)] = d3dur

        # Add pain and money levels for anticipation and shock (10)
        for lev, val in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            pon = np.where((events.painfirst == 1)
                           & ((events.pain_rank10l == val)
                              | (events.pain_rank10l == val)),
                           events['offer1_onset'],
                           9999)
            mon = np.where((events.painfirst == 0)
                           & ((events.money_rank10l == val)
                              | (events.money_rank10l == val)),
                           events['offer1_onset'],
                           9999)
            shck = np.where(events.pain_rank10l == val,
                            events['shock_onset_accept'],
                            9999)
            d3ons = np.where(events.pain_rank10l == val,
                             events['delay3_onset_accept'],
                             9999)
            d3dur = np.where(events.pain_rank10l == val,
                             events['delay3_duration_accept'],
                             9999)

            events['o1_ons_money_10l_' + str(lev + 1)] = mon
            events['o1_ons_pain_10l_' + str(lev + 1)] = pon
            events['shock_ons_accept_10l_' + str(lev + 1)] = shck
            events['delay3_ons_accept_10l_' + str(lev + 1)] = d3ons
            events['delay3_dur_accept_10l_' + str(lev + 1)] = d3dur
        # Add run
        events['run'] = idx + 1

        # Add offer2 as a a combination of pain offer levels (10)
        for pa in range(1, 11):
            for m in range(1, 11):
                ons_off2 = ['o2_pain' + str((int(pa)))
                            + '_money' + str(int(m)) + '_ons10l'][0]
                dur_off2 = ['o2_pain' + str((int(pa)))
                            + '_money' + str(int(m)) + '_dur10l'][0]

                events[ons_off2] = [float(9999)] * len(events)
                events[dur_off2] = [float(9999)] * len(events)

        for pa in range(1, 6):
            for m in range(1, 6):

                ons_off25l = ['o2_pain' + str((int(pa)))
                              + '_money' + str(int(m)) + '_ons5l'][0]
                dur_off25l = ['o2_pain' + str((int(pa)))
                              + '_money' + str(int(m)) + '_dur5l'][0]

                events[ons_off25l] = [float(9999)] * len(events)
                events[dur_off25l] = [float(9999)] * len(events)

        for index, row in events.iterrows():
            # Get the combination of levels for offer2
            ons_off2 = ['o2_pain' + str(int(row.pain_rank10l))
                        + '_money' + str(int(row.money_rank10l))
                        + '_ons10l'][0]
            dur_off2 = ['o2_pain' + str(int(row.pain_rank10l))
                        + '_money' + str(int(row.money_rank10l))
                        + '_dur10l'][0]

            ons_off25l = ['o2_pain' + str(int(row.pain_rank5l))
                          + '_money' + str(int(row.money_rank5l))
                          + '_ons5l'][0]
            dur_off25l = ['o2_pain' + str(int(row.pain_rank5l))
                          + '_money' + str(int(row.money_rank5l))
                          + '_dur5l'][0]

            events[ons_off2][index] = row.offer2_onset
            events[dur_off2][index] = row.offer2_duration
            events[ons_off25l][index] = row.offer2_onset
            events[dur_off25l][index] = row.offer2_duration

        # Save
        events.to_csv(osj(bidssubfold, 'func', bidssub + '_task-npng_'
                          + 'run-' + str(run) + '_events.tsv'), sep='\t',
                      index=False)

