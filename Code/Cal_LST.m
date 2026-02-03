%% Generate 30 m Daily All-Weather LST

% Author        : Zhao Junli
% Last Modified : 2026-01-19
%
% Description:
% This script generates 30 m daily all-weather LST
% by combining:
% 1) Reconstructed 30 m residual term (from XGBoost prediction)
% 2) Downscaled 30 m ATC model component (computed from alpha/beta + NDVI)

clc;
clear;
warning off;


%% PART 0. USER SETTINGS 

InputPath_Y_30m     = 'H:\WL_30m_LST_V2\Process\XGBoost_30m\Results_30m\N15000\ALL\';
InputPath_X_30m     = 'H:\WL_30m_LST_V2\Process\XGBoost_30m\Input\';

InputFile_Mask_30m  = 'H:\WL_30m_LST\Process\Auxiliary_Data\mask_30m.mat';
InputFile_AlphaPred = 'H:\WL_30m_LST_V2\Process\ATC_Downscaling\Predict\alpha\N15000\A6\alpha_Y_Compare.xlsx';
InputFile_BetaPred  = 'H:\WL_30m_LST_V2\Process\ATC_Downscaling\Predict\beta\N15000\B6\beta_Y_Compare.xlsx';
InputFile_CLCD_30m  = 'H:\WL_30m_LST\Process\Auxiliary_Data\CLCD_30m_aligned.tif';

OutputPath_LST_30m  = 'H:\WL_30m_LST_V2\Process\LST\N15000ALLA6B6\';

CoordRefSysCode_30m = 32648;


%% PART 1. LOAD MASK / ALPHA / BETA / REFERENCE

MaskData_30m = importdata(InputFile_Mask_30m);

AlphaMap_30m = nan(size(MaskData_30m));
BetaMap_30m  = nan(size(MaskData_30m));

AlphaTable_30m = readtable(InputFile_AlphaPred, 'Sheet', 'Predict_30m');
AlphaMap_30m(MaskData_30m) = AlphaTable_30m.alpha_pred;

BetaTable_30m = readtable(InputFile_BetaPred, 'Sheet', 'Predict_30m');
BetaMap_30m(MaskData_30m) = BetaTable_30m.beta_pred;

[CLCD_30m, RefR_30m] = readgeoraster(InputFile_CLCD_30m);

%% 

FileList_Y_30m = dir(fullfile(InputPath_Y_30m, '*.csv'));
FileNames_Y_30m = {FileList_Y_30m.name}';
nFiles_Y_30m = numel(FileNames_Y_30m);

for iFile = 1:nFiles_Y_30m

    disp(iFile)

    ResMap_30m  = nan(size(MaskData_30m));
    NDVIMap_30m = nan(size(MaskData_30m));
    nATCMap_30m = nan(size(MaskData_30m));

    Y_FileName = FileNames_Y_30m{iFile};
    Y_parts = split(Y_FileName, {'_', '.'});
    doy = str2double(Y_parts(3));

    PredTable_Y_30m = readtable(fullfile(InputPath_Y_30m, Y_FileName));
    ResMap_30m(MaskData_30m) = PredTable_Y_30m.Predicted_Val;

    X_FileName = sprintf('X_30m_%03d.csv', doy);
    XTable_30m = readtable(fullfile(InputPath_X_30m, X_FileName));
    NDVIMap_30m(MaskData_30m) = XTable_30m.NDVI;

    for a = 1:844
        for b = 1:749
            if ~isnan(AlphaMap_30m(a,b))
                nATCMap_30m(a,b) = AlphaMap_30m(a,b) + ...
                    (BetaMap_30m(a,b)) ./ (1 + exp(NDVIMap_30m(a,b))) .* ...
                    sin((2*pi.*(doy-79))./365);
            end
        end
    end

    LSTMap_30m = ResMap_30m + nATCMap_30m;

    LSTMap_30m(LSTMap_30m < 100) = NaN;
    LSTMap_30m(LSTMap_30m == 0)  = NaN;

    OutName_LST = sprintf('LST_%03d.tif', doy);

%     geotiffwrite(fullfile(OutputPath_LST_30m, OutName_LST), ...
%         LSTMap_30m, RefR_30m, 'CoordRefSysCode', CoordRefSysCode_30m);

end