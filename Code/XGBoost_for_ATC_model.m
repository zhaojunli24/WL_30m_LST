%% XGBoost for ATC model (Training 1 km + Prediction 30 m)
%
% Author        : Zhao Junli
% Last Modified : 2026-01-29
%
% Description:
% This script prepares feature matrices for ATC downscaling, including:
% 1) 1 km training samples for Alpha/Beta regression (X features + y labels)
% 2) 30 m prediction input features for Alpha/Beta inference (X features only)

clc;
clear;
warning off;


%% PART 0. USER SETTINGS 

N_max = 15000;

InputPath_SR_1km        = 'H:\WL_30m_LST\Process\SR_1km\';
InputPath_qeff_1km      = 'H:\WL_30m_LST\Process\q_effective_ACS_1km\';
InputPath_Alpha_1km     = 'H:\WL_30m_LST_V2\Process\ATC\Alpha_1km_2022.tif';
InputPath_Beta_1km      = 'H:\WL_30m_LST_V2\Process\ATC\Beta_1km_2022.tif';
InputPath_NDVI_MOD13A2  = 'H:\WL_30m_LST\Data\MOD13A2\';

InputPath_SR_30m        = 'H:\WL_30m_LST\Process\SR_30m\';
InputPath_qeff_30m      = 'H:\WL_30m_LST\Process\q_effective_ACS_30m\';

OutputPath_TrainRoot    = 'H:\WL_30m_LST_V2\Process\ATC_Downscaling\Train_v2';
OutputPath_PredictRoot  = 'H:\WL_30m_LST_V2\Process\ATC_Downscaling\Predict_v2';

AuxPath_Aspect_1km        = 'H:\WL_30m_LST\Process\Auxiliary_Data\Aspect_1km_aligned.tif';
AuxPath_ELV_1km         = 'H:\WL_30m_LST\Process\Auxiliary_Data\ELV_1km_aligned.tif';
AuxPath_Slope_1km       = 'H:\WL_30m_LST\Process\Auxiliary_Data\SLOPE_1km_aligned.tif';
AuxPath_SSP_1km         = 'H:\WL_30m_LST\Process\Auxiliary_Data\SSP_1km_aligned.tif';
AuxPath_SVF_1km         = 'H:\WL_30m_LST\Process\Auxiliary_Data\SVF_1km_aligned.tif';
AuxPath_RSRI_1km        = 'F:\My_Project\WL_30m_LST_17to20\Data\DEM\RSRI_1km.tif';

AuxPath_Aspect_30m        = 'H:\WL_30m_LST\Process\Auxiliary_Data\Aspect_30m_aligned.tif';
AuxPath_DEM_30m         = 'H:\WL_30m_LST\Process\Auxiliary_Data\DEM_30m_aligned.tif';
AuxPath_Slope_30m       = 'H:\WL_30m_LST\Process\Auxiliary_Data\SLOPE_30m_aligned.tif';
AuxPath_SSP_30m         = 'H:\WL_30m_LST\Process\Auxiliary_Data\SSP_30m_aligned.tif';
AuxPath_SVF_30m         = 'H:\WL_30m_LST\Process\Auxiliary_Data\SVF_30m_aligned.tif';
AuxPath_RSRI_30m        = 'H:\WL_30m_LST\Process\Auxiliary_Data\RSRI_30m_aligned.tif';

MaskSavePath_30m        = 'H:\WL_30m_LST\Process\Auxiliary_Data\mask_30m.mat';
InputFile_CenterRef_30m = 'H:\WL_30m_LST\Process\q_effective_ACS_30m\q_effective_20220101.tif';

RefYear_NDVI = 2022;
TotalDays    = 365;

feature_names_alpha = { ...
    'B','G','R','NIR','SWIR', ...
    'NDVI', 'qeff', ...
    'elv','slope','aspect','SVF','SSP','lat','RSRI'};

feature_names_beta = { ...
    feature_names_alpha{:}, ...
    'NDVI_range','qeff_log_range'};


%% PART 2. BUILD 1 KM TRAINING CSV (ALPHA / BETA)

Alpha_1km = readgeoraster(InputPath_Alpha_1km);
Beta_1km  = readgeoraster(InputPath_Beta_1km);

[Elv_1km,  R0_1km]      = readgeoraster(AuxPath_ELV_1km);
[Slope_1km,~]      = readgeoraster(AuxPath_Slope_1km);
[SSP_1km,  ~]      = readgeoraster(AuxPath_SSP_1km);
[SVF_1km,  ~]      = readgeoraster(AuxPath_SVF_1km);
[Aspect_1km,~]     = readgeoraster(AuxPath_Aspect_1km);
[RSRI_1km, ~]      = readgeoraster(AuxPath_RSRI_1km);

Xw_1km = R0_1km.XWorldLimits(1) + 0.5 * R0_1km.CellExtentInWorldX : ...
         R0_1km.CellExtentInWorldX : ...
         R0_1km.XWorldLimits(2) - 0.5 * R0_1km.CellExtentInWorldX;

Yw_1km = R0_1km.YWorldLimits(2) - 0.5 * R0_1km.CellExtentInWorldY : ...
        -R0_1km.CellExtentInWorldY : ...
         R0_1km.YWorldLimits(1) + 0.5 * R0_1km.CellExtentInWorldY;

[Xgrid_1km, Ygrid_1km] = meshgrid(Xw_1km, Yw_1km);
[lat_1km, lon_1km] = projinv(R0_1km.ProjectedCRS, Xgrid_1km, Ygrid_1km);

qeff_files_1km = dir(fullfile(InputPath_qeff_1km, '*.tif'));
n_qeff_1km = numel(qeff_files_1km);

qeff_doy_1km  = zeros(n_qeff_1km, 1);
qeff_data_1km = cell(n_qeff_1km, 1);

for i = 1:n_qeff_1km
    fprintf('q_effective_1km | %d / %d\n', i, n_qeff_1km);

    fname = qeff_files_1km(i).name;
    date_str = regexp(fname, '\d{8}', 'match', 'once');
    t_date = datetime(date_str, 'InputFormat', 'yyyyMMdd');
    qeff_doy_1km(i) = day(t_date, 'dayofyear');

    tmp = readgeoraster(fullfile(InputPath_qeff_1km, fname));
    qeff_data_1km{i} = double(tmp);
end

SR_files_1km = dir(fullfile(InputPath_SR_1km, '*.tif'));
n_SR_1km = numel(SR_files_1km);

SR_doy_1km = zeros(n_SR_1km, 1);

SR_B_1km    = cell(n_SR_1km, 1);
SR_G_1km    = cell(n_SR_1km, 1);
SR_R_1km    = cell(n_SR_1km, 1);
SR_NIR_1km  = cell(n_SR_1km, 1);
SR_SWIR_1km = cell(n_SR_1km, 1);

NDSI_1km = cell(n_SR_1km, 1);
NDWI_1km = cell(n_SR_1km, 1);

for i = 1:n_SR_1km
    fprintf('SR_1km | %d / %d\n', i, n_SR_1km);

    fname = SR_files_1km(i).name;
    date_str = regexp(fname, '\d{8}', 'match', 'once');
    t_date = datetime(date_str, 'InputFormat', 'yyyyMMdd');
    SR_doy_1km(i) = day(t_date, 'dayofyear');

    tmp = readgeoraster(fullfile(InputPath_SR_1km, fname));
    tmp = double(tmp);
    tmp(tmp == -9999) = NaN;

    B    = tmp(:,:,1) / 10000;
    G    = tmp(:,:,2) / 10000;
    R    = tmp(:,:,3) / 10000;
    NIR  = tmp(:,:,4) / 10000;
    SWIR = tmp(:,:,5) / 10000;

    SR_B_1km{i}    = B;
    SR_G_1km{i}    = G;
    SR_R_1km{i}    = R;
    SR_NIR_1km{i}  = NIR;
    SR_SWIR_1km{i} = SWIR;
end

NDVI_files_1km = dir(fullfile(InputPath_NDVI_MOD13A2, '*.tif'));
n_NDVI_1km = numel(NDVI_files_1km);

NDVI_dates_1km = datetime.empty(n_NDVI_1km, 0);
NDVI_doy_1km   = zeros(n_NDVI_1km, 1);

for t = 1:n_NDVI_1km

    fprintf('NDVI_1km | %d / %d\n', t, n_NDVI_1km);

    fname = NDVI_files_1km(t).name;

    date_token = regexp(fname, '(\d{4})_(\d{2})_(\d{2})', 'tokens', 'once');
    NDVI_dates_1km(t) = datetime( ...
        str2double(date_token{1}), ...
        str2double(date_token{2}), ...
        str2double(date_token{3}) );

    NDVI_doy_1km(t) = days(NDVI_dates_1km(t) - datetime(RefYear_NDVI,1,1)) + 1;

    [NDVI_raw, ~] = readgeoraster(fullfile(InputPath_NDVI_MOD13A2, fname));
    NDVI_raw = double(NDVI_raw);

    NDVI_val   = NDVI_raw(:,:,1) * 0.0001;
    DetailedQA = NDVI_raw(:,:,3);
    SummaryQA  = NDVI_raw(:,:,12);

    if t == 1
        [nRow_1km, nCol_1km] = size(NDVI_val);
        NDVI_ts_1km = nan(nRow_1km, nCol_1km, n_NDVI_1km);
    end

    mask_valid = ~isnan(NDVI_val);
    NDVI_tmp = NDVI_val;

    if any(mask_valid(:))

        qa = DetailedQA(mask_valid);
        qa_bin = dec2bin(qa, 16);

        vi_quality = bin2dec(qa_bin(:,15:16));
        vi_use     = bin2dec(qa_bin(:,11:14));

        sumQA = SummaryQA(mask_valid);

        good = (sumQA == 0) | ...
               (sumQA == 1 & vi_quality <= 1 & vi_use <= 1);

        mask_good = false(nRow_1km, nCol_1km);
        mask_good(mask_valid) = good;

        NDVI_tmp(~mask_good) = NaN;

    else
        NDVI_tmp(:) = NaN;
    end

    NDVI_ts_1km(:,:,t) = NDVI_tmp;
end

DOY_all = (1:TotalDays)';
NDVI_daily_1km = nan(nRow_1km, nCol_1km, TotalDays);

for r = 1:nRow_1km
    for c = 1:nCol_1km

        ts_ndvi = squeeze(NDVI_ts_1km(r, c, :));

        if all(isnan(ts_ndvi))
            continue;
        end

        valid = ~isnan(ts_ndvi);
        if sum(valid) < 4
            continue;
        end

        doy_v  = NDVI_doy_1km(valid);
        ndvi_v = ts_ndvi(valid);

        ndvi_interp = interp1(doy_v, ndvi_v, DOY_all, 'linear', NaN);

        ndvi_interp(DOY_all < min(doy_v)) = ndvi_v(1);
        ndvi_interp(DOY_all > max(doy_v)) = ndvi_v(end);

        NDVI_daily_1km(r, c, :) = ndvi_interp;
    end
end


SR_B_stack_1km    = cat(3, SR_B_1km{:});
SR_G_stack_1km    = cat(3, SR_G_1km{:});
SR_R_stack_1km    = cat(3, SR_R_1km{:});
SR_NIR_stack_1km  = cat(3, SR_NIR_1km{:});
SR_SWIR_stack_1km = cat(3, SR_SWIR_1km{:});

SR_B_mean_1km    = nanmean(SR_B_stack_1km,    3);
SR_G_mean_1km    = nanmean(SR_G_stack_1km,    3);
SR_R_mean_1km    = nanmean(SR_R_stack_1km,    3);
SR_NIR_mean_1km  = nanmean(SR_NIR_stack_1km,  3);
SR_SWIR_mean_1km = nanmean(SR_SWIR_stack_1km, 3);

NDVI_mean_1km = nanmean(NDVI_daily_1km, 3);

NDVI_range_1km = nanmax(NDVI_daily_1km, [], 3) - nanmin(NDVI_daily_1km, [], 3);

qeff_3d_1km = cat(3, qeff_data_1km{:});
qeff_3d_1km(qeff_3d_1km <= 0) = NaN;

qeff_mean_1km      = nanmean(qeff_3d_1km, 3);
log_qeff_1km       = log10(qeff_3d_1km);
qeff_log_range_1km = nanmax(log_qeff_1km, [], 3) - nanmin(log_qeff_1km, [], 3);

[center_data_30m, R_center_30m] = readgeoraster(InputFile_CenterRef_30m);

[rows_c_30m, cols_c_30m] = size(center_data_30m);
[x_center, y_center] = pix2map(R_center_30m, round(rows_c_30m/2), round(cols_c_30m/2));

[center_r_1km, center_c_1km] = map2pix(R0_1km, x_center, y_center);
center_r_1km = round(center_r_1km);
center_c_1km = round(center_c_1km);

[rows_1km, cols_1km] = size(Elv_1km);
[row_idx_1km, col_idx_1km] = ndgrid(1:rows_1km, 1:cols_1km);

dist_map_1km  = sqrt((row_idx_1km - center_r_1km).^2 + (col_idx_1km - center_c_1km).^2);
dist_flat_1km = dist_map_1km(:);

valid_mask_1km = (~isnan(SR_B_stack_1km(:,:,1))) & (~isnan(NDVI_mean_1km));
all_idx_1km = find(valid_mask_1km);

[~, order_1km] = sort(dist_flat_1km(all_idx_1km));
all_idx_1km = all_idx_1km(order_1km);

idx_train_1km = all_idx_1km(1:min(N_max, numel(all_idx_1km)));

out_dir_train = fullfile(OutputPath_TrainRoot, ['N' num2str(N_max)]);
if ~exist(out_dir_train, 'dir')
    mkdir(out_dir_train);
end

Alpha_X_1km = [ ...
    SR_B_mean_1km(idx_train_1km), SR_G_mean_1km(idx_train_1km), SR_R_mean_1km(idx_train_1km), ...
    SR_NIR_mean_1km(idx_train_1km), SR_SWIR_mean_1km(idx_train_1km), ...
    NDVI_mean_1km(idx_train_1km), ...
    qeff_mean_1km(idx_train_1km), ...
    Elv_1km(idx_train_1km), Slope_1km(idx_train_1km), Aspect_1km(idx_train_1km), ...
    SVF_1km(idx_train_1km), SSP_1km(idx_train_1km), ...
    lat_1km(idx_train_1km), RSRI_1km(idx_train_1km) ];

Alpha_y_1km = Alpha_1km(idx_train_1km);

Alpha_X_tbl_1km = array2table(Alpha_X_1km, 'VariableNames', feature_names_alpha);
Alpha_y_tbl_1km = table(Alpha_y_1km, 'VariableNames', {'Alpha'});

writetable(Alpha_X_tbl_1km, fullfile(out_dir_train, 'Alpha_X.csv'));
writetable(Alpha_y_tbl_1km, fullfile(out_dir_train, 'Alpha_y.csv'));

Beta_X_1km = [ ...
    Alpha_X_1km, ...
    NDVI_range_1km(idx_train_1km), ...
    qeff_log_range_1km(idx_train_1km) ];

Beta_y_1km = Beta_1km(idx_train_1km);

Beta_X_tbl_1km = array2table(Beta_X_1km, 'VariableNames', feature_names_beta);
Beta_y_tbl_1km = table(Beta_y_1km, 'VariableNames', {'Beta'});

writetable(Beta_X_tbl_1km, fullfile(out_dir_train, 'Beta_X.csv'));
writetable(Beta_y_tbl_1km, fullfile(out_dir_train, 'Beta_y.csv'));

%% PART 3. BUILD 30 M PREDICTION CSV (ALPHA / BETA)

[Elv_30m,  R0_30m]      = readgeoraster(AuxPath_DEM_30m);
[Slope_30m,~]      = readgeoraster(AuxPath_Slope_30m);
[SSP_30m,  ~]      = readgeoraster(AuxPath_SSP_30m);
[SVF_30m,  ~]      = readgeoraster(AuxPath_SVF_30m);
[Aspect_30m,~]     = readgeoraster(AuxPath_Aspect_30m);
[RSRI_30m, ~]      = readgeoraster(AuxPath_RSRI_30m);

Xw_30m = R0_30m.XWorldLimits(1) + 0.5 * R0_30m.CellExtentInWorldX : ...
         R0_30m.CellExtentInWorldX : ...
         R0_30m.XWorldLimits(2) - 0.5 * R0_30m.CellExtentInWorldX;

Yw_30m = R0_30m.YWorldLimits(2) - 0.5 * R0_30m.CellExtentInWorldY : ...
        -R0_30m.CellExtentInWorldY : ...
         R0_30m.YWorldLimits(1) + 0.5 * R0_30m.CellExtentInWorldY;

[Xgrid_30m, Ygrid_30m] = meshgrid(Xw_30m, Yw_30m);
[lat_30m, lon_30m] = projinv(R0_30m.ProjectedCRS, Xgrid_30m, Ygrid_30m);

qeff_files_30m = dir(fullfile(InputPath_qeff_30m, '*.tif'));
n_qeff_30m = numel(qeff_files_30m);

qeff_doy_30m  = zeros(n_qeff_30m, 1);
qeff_data_30m = cell(n_qeff_30m, 1);

for i = 1:n_qeff_30m
    fprintf('q_effective_30m | %d / %d\n', i, n_qeff_30m);

    fname = qeff_files_30m(i).name;
    date_str = regexp(fname, '\d{8}', 'match', 'once');
    t_date = datetime(date_str, 'InputFormat', 'yyyyMMdd');
    qeff_doy_30m(i) = day(t_date, 'dayofyear');

    tmp = readgeoraster(fullfile(InputPath_qeff_30m, fname));
    qeff_data_30m{i} = double(tmp);
end

SR_files_30m = dir(fullfile(InputPath_SR_30m, '*.tif'));
n_SR_30m = numel(SR_files_30m);

SR_doy_30m = zeros(n_SR_30m, 1);

SR_B_30m    = cell(n_SR_30m, 1);
SR_G_30m    = cell(n_SR_30m, 1);
SR_R_30m    = cell(n_SR_30m, 1);
SR_NIR_30m  = cell(n_SR_30m, 1);
SR_SWIR_30m = cell(n_SR_30m, 1);

NDVI_30m = cell(n_SR_30m, 1);
NDSI_30m = cell(n_SR_30m, 1);
NDWI_30m = cell(n_SR_30m, 1);

for i = 1:n_SR_30m
    fprintf('SR_30m | %d / %d\n', i, n_SR_30m);

    fname = SR_files_30m(i).name;
    date_str = regexp(fname, '\d{8}', 'match', 'once');
    t_date = datetime(date_str, 'InputFormat', 'yyyyMMdd');
    SR_doy_30m(i) = day(t_date, 'dayofyear');

    tmp = readgeoraster(fullfile(InputPath_SR_30m, fname));
    tmp = double(tmp);
    tmp(tmp == -9999) = NaN;

    B    = tmp(:,:,1) / 10000;
    G    = tmp(:,:,2) / 10000;
    R    = tmp(:,:,3) / 10000;
    NIR  = tmp(:,:,4) / 10000;
    SWIR = tmp(:,:,5) / 10000;

    SR_B_30m{i}    = B;
    SR_G_30m{i}    = G;
    SR_R_30m{i}    = R;
    SR_NIR_30m{i}  = NIR;
    SR_SWIR_30m{i} = SWIR;

    ndvi = (NIR - R) ./ (NIR + R);

    ndvi = ndvi * 0.70956 + 0.18506;
    ndvi(ndvi > 0.99) = 0.99;
    ndvi(ndvi < -0.1) = -0.2;

    NDVI_30m{i} = ndvi;
end

SR_B_stack_30m    = cat(3, SR_B_30m{:});
SR_G_stack_30m    = cat(3, SR_G_30m{:});
SR_R_stack_30m    = cat(3, SR_R_30m{:});
SR_NIR_stack_30m  = cat(3, SR_NIR_30m{:});
SR_SWIR_stack_30m = cat(3, SR_SWIR_30m{:});

SR_B_mean_30m    = nanmean(SR_B_stack_30m,    3);
SR_G_mean_30m    = nanmean(SR_G_stack_30m,    3);
SR_R_mean_30m    = nanmean(SR_R_stack_30m,    3);
SR_NIR_mean_30m  = nanmean(SR_NIR_stack_30m,  3);
SR_SWIR_mean_30m = nanmean(SR_SWIR_stack_30m, 3);

NDVI_stack_30m = cat(3, NDVI_30m{:});

NDVI_mean_30m = nanmean(NDVI_stack_30m, 3);

NDVI_range_30m = nanmax(NDVI_stack_30m, [], 3) - nanmin(NDVI_stack_30m, [], 3);

qeff_3d_30m = cat(3, qeff_data_30m{:});
qeff_3d_30m(qeff_3d_30m <= 0) = NaN;

qeff_mean_30m      = nanmean(qeff_3d_30m, 3);
log_qeff_30m       = log10(qeff_3d_30m);
qeff_log_range_30m = nanmax(log_qeff_30m, [], 3) - nanmin(log_qeff_30m, [], 3);

mask_30m = ~isnan(qeff_data_30m{1});
idx_30m  = find(mask_30m);

fprintf('ATC 30m valid pixels: %d\n', numel(idx_30m));

save(MaskSavePath_30m, 'mask_30m');

out_dir_pred = OutputPath_PredictRoot;
if ~exist(out_dir_pred, 'dir')
    mkdir(out_dir_pred);
end

Alpha_X_30m = [ ...
    SR_B_mean_30m(idx_30m), SR_G_mean_30m(idx_30m), SR_R_mean_30m(idx_30m), ...
    SR_NIR_mean_30m(idx_30m), SR_SWIR_mean_30m(idx_30m), ...
    NDVI_mean_30m(idx_30m), ...
    qeff_mean_30m(idx_30m), ...
    Elv_30m(idx_30m), Slope_30m(idx_30m), Aspect_30m(idx_30m), ...
    SVF_30m(idx_30m), SSP_30m(idx_30m), ...
    lat_30m(idx_30m), RSRI_30m(idx_30m) ];

Alpha_tbl_30m = array2table(Alpha_X_30m, 'VariableNames', feature_names_alpha);
writetable(Alpha_tbl_30m, fullfile(out_dir_pred, 'Alpha_X_30m.csv'));

Beta_X_30m = [ ...
    Alpha_X_30m, ...
    NDVI_range_30m(idx_30m), ...
    qeff_log_range_30m(idx_30m) ];

Beta_tbl_30m = array2table(Beta_X_30m, 'VariableNames', feature_names_beta);
writetable(Beta_tbl_30m, fullfile(out_dir_pred, 'Beta_X_30m.csv'));
