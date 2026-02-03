%% Create XGBoost Downscaling Samples (Training 1 km + Prediction 30 m)
%
% Author        : Zhao Junli
% Last Modified : 2026-01-29
%
% Description:
% This script prepares feature matrices for XGBoost-based reconstruction of
% 30 m LST residuals, including:
% 1) Training samples at 1 km resolution (X features + Y residual labels)
% 2) Prediction input features at 30 m resolution (X features only)
%
% Notes:
% 1) Clear-sky valid pixels are selected outward layer-by-layer from the
%    study-area center, with the maximum number of exported training samples
%    per day limited to N_max = 15000.
% 2) The distance of the farthest selected pixel from the center is used as
%    the dynamic window radius for daily sample selection.

clc;
clear;
warning off;


%% PART 0. USER SETTINGS 

TOTAL_DAYS = 365;     % Number of days in one year
REF_YEAR   = 2022;    % Reference year used for MOD13A2 DOY indexing

% TRAINING PATHS
InputPath_Residuals_1km  = 'H:\WL_30m_LST_V2\Process\LST_residual_1km\';
InputPath_SR_1km         = 'H:\WL_30m_LST\Process\SR_1km\';
InputPath_qEffective_1km = 'H:\WL_30m_LST\Process\q_effective_ACS_1km\';
InputPath_NDVI_1km       = 'H:\WL_30m_LST\Data\MOD13A2\';
InputPath_SnowFlag_1km   = 'H:\WL_30m_LST\Process\Snow_flag_1km\';

InputFile_CLCD_1km   = 'H:\WL_30m_LST\Process\Auxiliary_Data\CLCD_1km_aligned.tif';
InputFile_Elv_1km    = 'H:\WL_30m_LST\Process\Auxiliary_Data\ELV_1km_aligned.tif';
InputFile_Slope_1km  = 'H:\WL_30m_LST\Process\Auxiliary_Data\SLOPE_1km_aligned.tif';
InputFile_SSP_1km    = 'H:\WL_30m_LST\Process\Auxiliary_Data\SSP_1km_aligned.tif';
InputFile_SVF_1km    = 'H:\WL_30m_LST\Process\Auxiliary_Data\SVF_1km_aligned.tif';
InputFile_Aspect_1km = 'H:\WL_30m_LST\Process\Auxiliary_Data\ASPECT_1km_aligned.tif';

InputFile_CenterRef_30m = 'H:\WL_30m_LST\Process\q_effective_ACS_30m\q_effective_20220101.tif';

OutputPath_TrainRoot_1km = 'H:\WL_30m_LST_V2\Process\XGBoost_1km\Train';

% PREDICTION (30 m) PATHS
InputPath_SR_30m         = 'H:\WL_30m_LST\Process\SR_30m\';
InputPath_qEffective_30m = 'H:\WL_30m_LST\Process\q_effective_ACS_30m\';
InputPath_SnowFlag_30m   = 'H:\WL_30m_LST\Process\Snow_flag_30m\';

InputFile_CLCD_30m   = 'H:\WL_30m_LST\Process\Auxiliary_Data\CLCD_30m_aligned.tif';
InputFile_Elv_30m    = 'H:\WL_30m_LST\Process\Auxiliary_Data\DEM_30m_aligned.tif';
InputFile_Slope_30m  = 'H:\WL_30m_LST\Process\Auxiliary_Data\SLOPE_30m_aligned.tif';
InputFile_SSP_30m    = 'H:\WL_30m_LST\Process\Auxiliary_Data\SSP_30m_aligned.tif';
InputFile_SVF_30m    = 'H:\WL_30m_LST\Process\Auxiliary_Data\SVF_30m_aligned.tif';
InputFile_Aspect_30m = 'H:\WL_30m_LST\Process\Auxiliary_Data\ASPECT_30m_aligned.tif';

OutputPath_PredictRoot_30m = 'H:\WL_30m_LST_V2\Process\XGBoost_30m\Input';

% TRAINING SAMPLE SETTINGS
N_list_1km = 15000;  % can be scalar or vector, e.g. [1000 5000 15000]

% Feature names (must be consistent between training and prediction)
feature_names = { ...
    'B','G','R','NIR','SWIR', ...
    'NDVI','NDSI','NDWI', ...
    'qeff','snow', ...
    'elv','slope','aspect','SVF','SSP','lat','SAI','CLCD'};

%% PART 1. READ RESIDUAL MAPS (1 km)

res_files_1km = dir(fullfile(InputPath_Residuals_1km, 'residuals_*.tif'));
n_res_1km     = numel(res_files_1km);

res_doy_1km  = zeros(n_res_1km, 1);
res_data_1km = cell(n_res_1km, 1);
R_res_1km    = cell(n_res_1km, 1);

[~, R0_1km] = readgeoraster(fullfile(InputPath_Residuals_1km, res_files_1km(1).name));

Xw_1km = R0_1km.XWorldLimits(1) + 0.5 * R0_1km.CellExtentInWorldX : ...
    R0_1km.CellExtentInWorldX : ...
    R0_1km.XWorldLimits(2) - 0.5 * R0_1km.CellExtentInWorldX;

Yw_1km = R0_1km.YWorldLimits(2) - 0.5 * R0_1km.CellExtentInWorldY : ...
   -R0_1km.CellExtentInWorldY : ...
    R0_1km.YWorldLimits(1) + 0.5 * R0_1km.CellExtentInWorldY;

[X_res_1km, Y_res_1km] = meshgrid(Xw_1km, Yw_1km);

for i = 1:n_res_1km

    fprintf('Residuals_1km | %d / %d\n', i, n_res_1km);

    fname = res_files_1km(i).name;

    doy_str        = regexp(fname, '\d{3}', 'match', 'once');
    res_doy_1km(i) = str2double(doy_str);

    [tmp, R]        = readgeoraster(fullfile(InputPath_Residuals_1km, fname));
    res_data_1km{i} = double(tmp);
    R_res_1km{i}    = R;

end


%% PART 2. READ AUXILIARY DATA (1 km)

[CLCD_1km,   ~] = readgeoraster(InputFile_CLCD_1km);
[Elv_1km,    ~] = readgeoraster(InputFile_Elv_1km);
[Slope_1km,  ~] = readgeoraster(InputFile_Slope_1km);
[SSP_1km,    ~] = readgeoraster(InputFile_SSP_1km);
[SVF_1km,    ~] = readgeoraster(InputFile_SVF_1km);
[Aspect_1km, ~] = readgeoraster(InputFile_Aspect_1km);

[lat_1km, lon_1km] = projinv(R0_1km.ProjectedCRS, X_res_1km, Y_res_1km);


%% PART 3. READ q_effective AND Snow_flag (1 km)

qeff_files_1km = dir(fullfile(InputPath_qEffective_1km, '*.tif'));
n_qeff_1km     = numel(qeff_files_1km);

qeff_doy_1km  = zeros(n_qeff_1km, 1);
qeff_data_1km = cell(n_qeff_1km, 1);

for i = 1:n_qeff_1km

    fprintf('q_effective_1km | %d / %d\n', i, n_qeff_1km);

    fname = qeff_files_1km(i).name;

    date_str         = regexp(fname, '\d{8}', 'match', 'once');
    t_date           = datetime(date_str, 'InputFormat', 'yyyyMMdd');
    qeff_doy_1km(i)  = day(t_date, 'dayofyear');

    [tmp, ~]          = readgeoraster(fullfile(InputPath_qEffective_1km, fname));
    qeff_data_1km{i}  = double(tmp);

end

snow_files_1km = dir(fullfile(InputPath_SnowFlag_1km, '*.tif'));
n_snow_1km     = numel(snow_files_1km);

snow_doy_1km  = zeros(n_snow_1km, 1);
snow_data_1km = cell(n_snow_1km, 1);

for i = 1:n_snow_1km

    fprintf('Snow_flag_1km | %d / %d\n', i, n_snow_1km);

    fname = snow_files_1km(i).name;

    date_str          = regexp(fname, '\d{8}', 'match', 'once');
    t_date            = datetime(date_str, 'InputFormat', 'yyyyMMdd');
    snow_doy_1km(i)   = day(t_date, 'dayofyear');

    [tmp, ~]          = readgeoraster(fullfile(InputPath_SnowFlag_1km, fname));
    snow_data_1km{i}  = double(tmp);

end


%% PART 4. READ MOD13 NDVI AND APPLY QA FILTERING (1 km)

NDVI_files_1km = dir(fullfile(InputPath_NDVI_1km, '*.tif'));
nTime_NDVI_1km = numel(NDVI_files_1km);

NDVI_dates_1km = datetime.empty(nTime_NDVI_1km, 0);
NDVI_doy_1km   = zeros(nTime_NDVI_1km, 1);

for t = 1:nTime_NDVI_1km

    fprintf('NDVI_1km | %d / %d\n', t, nTime_NDVI_1km);

    fname = NDVI_files_1km(t).name;

    date_token       = regexp(fname, '(\d{4})_(\d{2})_(\d{2})', 'tokens', 'once');
    NDVI_dates_1km(t) = datetime( ...
        str2double(date_token{1}), ...
        str2double(date_token{2}), ...
        str2double(date_token{3}) );

    NDVI_doy_1km(t) = days(NDVI_dates_1km(t) - datetime(REF_YEAR, 1, 1)) + 1;

    [NDVI_raw, ~] = readgeoraster(fullfile(InputPath_NDVI_1km, fname));
    NDVI_raw = double(NDVI_raw);

    NDVI_val   = NDVI_raw(:,:,1) * 0.0001;
    DetailedQA = NDVI_raw(:,:,3);
    SummaryQA  = NDVI_raw(:,:,12);

    if t == 1
        [nRow_1km, nCol_1km] = size(NDVI_val);
        NDVI_ts_1km = nan(nRow_1km, nCol_1km, nTime_NDVI_1km);
    end

    mask_valid = ~isnan(NDVI_val);
    NDVI_tmp   = NDVI_val;

    if any(mask_valid(:))

        qa     = DetailedQA(mask_valid);
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


%% PART 5. NDVI TEMPORAL INTERPOLATION TO DAILY SCALE (1 km)

NDVI_daily_1km = nan(nRow_1km, nCol_1km, n_res_1km);

for iRow = 1:nRow_1km

    fprintf('NDVI_interp_1km | row %d / %d\n', iRow, nRow_1km);

    for iCol = 1:nCol_1km

        ts_ndvi = squeeze(NDVI_ts_1km(iRow, iCol, :));

        if all(isnan(ts_ndvi))
            continue;
        end

        valid = ~isnan(ts_ndvi);
        if sum(valid) < 4
            continue;
        end

        doy_v  = NDVI_doy_1km(valid);
        ndvi_v = ts_ndvi(valid);

        ndvi_interp = interp1(doy_v, ndvi_v, res_doy_1km, 'linear');

        ndvi_interp(res_doy_1km < min(doy_v)) = ndvi_v(1);
        ndvi_interp(res_doy_1km > max(doy_v)) = ndvi_v(end);

        NDVI_daily_1km(iRow, iCol, :) = ndvi_interp;

    end
end



%% PART 6. READ SR AND COMPUTE SPECTRAL INDICES (1 km)

SR_files_1km = dir(fullfile(InputPath_SR_1km, '*.tif'));
n_SR_1km     = numel(SR_files_1km);

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

    date_str        = regexp(fname, '\d{8}', 'match', 'once');
    t_date          = datetime(date_str, 'InputFormat', 'yyyyMMdd');
    SR_doy_1km(i)   = day(t_date, 'dayofyear');

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

    ndsi = (G - SWIR) ./ (G + SWIR);
    ndwi = (G - NIR) ./ (G + NIR);

    NDSI_1km{i} = ndsi;
    NDWI_1km{i} = ndwi;

end

%% PART 7. TEMPORAL MATCHING TO RESIDUAL DOY (1 km)

SR_B_match_1km    = cell(n_res_1km, 1);
SR_G_match_1km    = cell(n_res_1km, 1);
SR_R_match_1km    = cell(n_res_1km, 1);
SR_NIR_match_1km  = cell(n_res_1km, 1);
SR_SWIR_match_1km = cell(n_res_1km, 1);

NDSI_match_1km = cell(n_res_1km, 1);
NDWI_match_1km = cell(n_res_1km, 1);

qeff_match_1km = cell(n_res_1km, 1);
snow_match_1km = cell(n_res_1km, 1);

for i = 1:n_res_1km

    doy = res_doy_1km(i);

    [~, idx_sr]   = min(abs(SR_doy_1km   - doy));
    [~, idx_qeff] = min(abs(qeff_doy_1km - doy));
    [~, idx_snow] = min(abs(snow_doy_1km - doy));

    SR_B_match_1km{i}    = SR_B_1km{idx_sr};
    SR_G_match_1km{i}    = SR_G_1km{idx_sr};
    SR_R_match_1km{i}    = SR_R_1km{idx_sr};
    SR_NIR_match_1km{i}  = SR_NIR_1km{idx_sr};
    SR_SWIR_match_1km{i} = SR_SWIR_1km{idx_sr};

    NDSI_match_1km{i} = NDSI_1km{idx_sr};
    NDWI_match_1km{i} = NDWI_1km{idx_sr};

    qeff_match_1km{i} = qeff_data_1km{idx_qeff};
    snow_match_1km{i} = snow_data_1km{idx_snow};


end


%% PART 8. LOCATE CENTER PIXEL IN RESIDUAL GRID

[center_data_30m, R_center_30m] = readgeoraster(InputFile_CenterRef_30m);

[rows_c_30m, cols_c_30m] = size(center_data_30m);
[x_center, y_center] = pix2map(R_center_30m, round(rows_c_30m/2), round(cols_c_30m/2));

[center_r_1km, center_c_1km] = map2pix(R0_1km, x_center, y_center);
center_r_1km = round(center_r_1km);
center_c_1km = round(center_c_1km);


%% PART 9. BUILD TRAINING SAMPLES (1 km)

[rows_1km, cols_1km] = size(res_data_1km{1});
[row_idx_1km, col_idx_1km] = ndgrid(1:rows_1km, 1:cols_1km);

dist_map_1km  = sqrt((row_idx_1km - center_r_1km).^2 + (col_idx_1km - center_c_1km).^2);
dist_flat_1km = dist_map_1km(:);

for iN = 1:length(N_list_1km)

    N_max = N_list_1km(iN);
    fprintf('N_max_1km | %d / %d\n', iN, length(N_list_1km));

    out_base_1km = fullfile(OutputPath_TrainRoot_1km, ['N' num2str(N_max)]);
    dir_X_1km    = fullfile(out_base_1km, 'X');
    dir_Y_1km    = fullfile(out_base_1km, 'Y');

    if ~exist(dir_X_1km, 'dir'); mkdir(dir_X_1km); end
    if ~exist(dir_Y_1km, 'dir'); mkdir(dir_Y_1km); end

    dist_max_list_1km = NaN(numel(res_data_1km), 1);

    for i = 1:numel(res_data_1km)

        fprintf('DOY_1km | %d / %d\n', i, numel(res_data_1km));

        doy     = res_doy_1km(i);
        doy_str = sprintf('%03d', doy);

        data_res_1km = res_data_1km{i};

        B_1km    = SR_B_match_1km{i};
        G_1km    = SR_G_match_1km{i};
        R_1km    = SR_R_match_1km{i};
        NIR_1km  = SR_NIR_match_1km{i};
        SWIR_1km = SR_SWIR_match_1km{i};

        NDVI_1km = NDVI_daily_1km(:,:,i);
        NDSI_1km = NDSI_match_1km{i};
        NDWI_1km = NDWI_match_1km{i};

        qeff_1km = qeff_match_1km{i};
        snow_1km = snow_match_1km{i};

        delta = 23.44 * sind(360/365 * (doy + 284));
        T = 3.63 + 104/15;
        H = 15 * (T - 12);

        cos_theta = sind(delta).*sind(lat_1km) + ...
                    cosd(delta).*cosd(lat_1km).*cosd(H);
        cos_theta = min(max(cos_theta, -1), 1);

        theta = acosd(cos_theta);
        theta(theta > 90) = 90;

        sin_theta = sind(theta);
        sin_theta(sin_theta == 0) = eps;

        omiga = asind((cosd(delta).*sind(H))./sin_theta) + 180;

        cos_SAI = cosd(Slope_1km).*cosd(theta) + ...
                  sind(Slope_1km).*sind(theta).*cosd(omiga - Aspect_1km);
        cos_SAI = min(max(cos_SAI, -1), 1);

        SAI_1km = acosd(cos_SAI);

        mask_total_1km = ~isnan(data_res_1km) & ~isnan(NIR_1km);
        mask_flat_1km  = mask_total_1km(:);

        [~, order] = sort(dist_flat_1km);
        valid_idx_1km  = order(mask_flat_1km(order));

        if numel(valid_idx_1km) > N_max
            valid_idx_1km = valid_idx_1km(1:N_max);
        end

        if isempty(valid_idx_1km)
            warning(['DOY ' doy_str ' has no valid pixels']);
            continue;
        end

        dist_max_list_1km(i) = dist_flat_1km(valid_idx_1km(end));

        X_1km = [ ...
            B_1km(valid_idx_1km), ...
            G_1km(valid_idx_1km), ...
            R_1km(valid_idx_1km), ...
            NIR_1km(valid_idx_1km), ...
            SWIR_1km(valid_idx_1km), ...
            NDVI_1km(valid_idx_1km), ...
            NDSI_1km(valid_idx_1km), ...
            NDWI_1km(valid_idx_1km), ...
            qeff_1km(valid_idx_1km), ...
            snow_1km(valid_idx_1km), ...
            Elv_1km(valid_idx_1km), ...
            Slope_1km(valid_idx_1km), ...
            Aspect_1km(valid_idx_1km), ...
            SVF_1km(valid_idx_1km), ...
            SSP_1km(valid_idx_1km), ...
            lat_1km(valid_idx_1km), ...
            SAI_1km(valid_idx_1km), ...
            CLCD_1km(valid_idx_1km) ...
        ];

        Y_1km = data_res_1km(valid_idx_1km);

        X_table_1km = array2table(X_1km, 'VariableNames', feature_names);
        Y_table_1km = table(Y_1km, 'VariableNames', {'residual'});

%         writetable(X_table_1km, fullfile(dir_X_1km, ['X_' doy_str '.csv']));
%         writetable(Y_table_1km, fullfile(dir_Y_1km, ['Y_' doy_str '.csv']));

    end

    dist_table_1km = table(res_doy_1km(:), dist_max_list_1km, ...
        'VariableNames', {'DOY','max_distance_pixel'});

%     writetable(dist_table_1km, fullfile(out_base_1km, 'max_distance.csv'));

end


%% PART 10. READ AUXILIARY DATA (30 m)

[CLCD_30m, R0_30m] = readgeoraster(InputFile_CLCD_30m);
[Elv_30m,  ~]      = readgeoraster(InputFile_Elv_30m);
[Slope_30m,~]      = readgeoraster(InputFile_Slope_30m);
[SSP_30m,  ~]      = readgeoraster(InputFile_SSP_30m);
[SVF_30m,  ~]      = readgeoraster(InputFile_SVF_30m);
[Aspect_30m,~]     = readgeoraster(InputFile_Aspect_30m);

Xw_30m = R0_30m.XWorldLimits(1) + 0.5 * R0_30m.CellExtentInWorldX : ...
    R0_30m.CellExtentInWorldX : ...
    R0_30m.XWorldLimits(2) - 0.5 * R0_30m.CellExtentInWorldX;

Yw_30m = R0_30m.YWorldLimits(2) - 0.5 * R0_30m.CellExtentInWorldY : ...
   -R0_30m.CellExtentInWorldY : ...
    R0_30m.YWorldLimits(1) + 0.5 * R0_30m.CellExtentInWorldY;

[X_res_30m, Y_res_30m] = meshgrid(Xw_30m, Yw_30m);

[lat_30m, lon_30m] = projinv(R0_30m.ProjectedCRS, X_res_30m, Y_res_30m);


%% PART 11. READ SR AND COMPUTE INDICES (30 m)

SR_files_30m = dir(fullfile(InputPath_SR_30m, '*.tif'));
n_SR_30m     = numel(SR_files_30m);

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

    date_str        = regexp(fname, '\d{8}', 'match', 'once');
    t_date          = datetime(date_str, 'InputFormat', 'yyyyMMdd');
    SR_doy_30m(i)   = day(t_date, 'dayofyear');

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
    ndsi = (G - SWIR) ./ (G + SWIR);
    ndwi = (G - NIR) ./ (G + NIR);

    ndvi = ndvi * 0.70956 + 0.18506;
    ndvi(ndvi > 0.99) = 0.99;
    ndvi(ndvi < -0.1) = -0.2;

    NDVI_30m{i} = ndvi;
    NDSI_30m{i} = ndsi;
    NDWI_30m{i} = ndwi;

end


%% PART 12. READ q_effective AND Snow_flag (30 m)

qeff_files_30m = dir(fullfile(InputPath_qEffective_30m, '*.tif'));
n_qeff_30m     = numel(qeff_files_30m);

qeff_doy_30m  = zeros(n_qeff_30m, 1);
qeff_data_30m = cell(n_qeff_30m, 1);

for i = 1:n_qeff_30m

    fprintf('q_effective_30m | %d / %d\n', i, n_qeff_30m);

    fname = qeff_files_30m(i).name;

    date_str         = regexp(fname, '\d{8}', 'match', 'once');
    t_date           = datetime(date_str, 'InputFormat', 'yyyyMMdd');
    qeff_doy_30m(i)  = day(t_date, 'dayofyear');

    [tmp, ~]          = readgeoraster(fullfile(InputPath_qEffective_30m, fname));
    qeff_data_30m{i}  = double(tmp);

end

snow_files_30m = dir(fullfile(InputPath_SnowFlag_30m, '*.tif'));
n_snow_30m     = numel(snow_files_30m);

snow_doy_30m  = zeros(n_snow_30m, 1);
snow_data_30m = cell(n_snow_30m, 1);

for i = 1:n_snow_30m

    fprintf('Snow_flag_30m | %d / %d\n', i, n_snow_30m);

    fname = snow_files_30m(i).name;

    date_str          = regexp(fname, '\d{8}', 'match', 'once');
    t_date            = datetime(date_str, 'InputFormat', 'yyyyMMdd');
    snow_doy_30m(i)   = day(t_date, 'dayofyear');

    [tmp, ~]          = readgeoraster(fullfile(InputPath_SnowFlag_30m, fname));
    snow_data_30m{i}  = double(tmp);

end


%% PART 13. TEMPORAL MATCHING TO DAILY DOY (30 m)

DOY_all_30m = (1:TOTAL_DAYS)';

SR_B_match_30m    = cell(TOTAL_DAYS, 1);
SR_G_match_30m    = cell(TOTAL_DAYS, 1);
SR_R_match_30m    = cell(TOTAL_DAYS, 1);
SR_NIR_match_30m  = cell(TOTAL_DAYS, 1);
SR_SWIR_match_30m = cell(TOTAL_DAYS, 1);

NDSI_match_30m = cell(TOTAL_DAYS, 1);
NDWI_match_30m = cell(TOTAL_DAYS, 1);

qeff_match_30m = cell(TOTAL_DAYS, 1);
snow_match_30m = cell(TOTAL_DAYS, 1);

for doy = 1:TOTAL_DAYS

    [~, idx_sr]   = min(abs(SR_doy_30m   - doy));
    [~, idx_qeff] = min(abs(qeff_doy_30m - doy));
    [~, idx_snow] = min(abs(snow_doy_30m - doy));

    SR_B_match_30m{doy}    = SR_B_30m{idx_sr};
    SR_G_match_30m{doy}    = SR_G_30m{idx_sr};
    SR_R_match_30m{doy}    = SR_R_30m{idx_sr};
    SR_NIR_match_30m{doy}  = SR_NIR_30m{idx_sr};
    SR_SWIR_match_30m{doy} = SR_SWIR_30m{idx_sr};

    NDSI_match_30m{doy} = NDSI_30m{idx_sr};
    NDWI_match_30m{doy} = NDWI_30m{idx_sr};

    qeff_match_30m{doy} = qeff_data_30m{idx_qeff};
    snow_match_30m{doy} = snow_data_30m{idx_snow};

end


%% PART 14. NDVI TEMPORAL INTERPOLATION TO DAILY SCALE (30 m)

[nr_30m, nc_30m] = size(NDVI_30m{1});
n_SR_30m         = numel(NDVI_30m);

NDVI_stack_30m = nan(nr_30m, nc_30m, n_SR_30m);
for i = 1:n_SR_30m
    NDVI_stack_30m(:,:,i) = NDVI_30m{i};
end

NDVI_match_30m = cell(TOTAL_DAYS, 1);
for d = 1:TOTAL_DAYS
    NDVI_match_30m{d} = nan(nr_30m, nc_30m);
end

for r = 1:nr_30m

    if mod(r, 100) == 0
        fprintf('NDVI_interp_30m | row %d / %d\n', r, nr_30m);
    end

    for c = 1:nc_30m

        ts = squeeze(NDVI_stack_30m(r, c, :));
        valid = ~isnan(ts) & ~isnan(SR_doy_30m);

        if sum(valid) < 2
            continue
        end

        ts_interp = interp1( ...
            SR_doy_30m(valid), ts(valid), ...
            DOY_all_30m, 'linear', NaN);

        for d = 1:TOTAL_DAYS
            NDVI_match_30m{d}(r, c) = ts_interp(d);
        end

    end
end

%% PART 15. EXPORT PREDICTION FEATURES X (30 m)

mask_30m = ~isnan(qeff_match_30m{1});
idx_30m  = find(mask_30m);

out_dir_30m = OutputPath_PredictRoot_30m;
if ~exist(out_dir_30m, 'dir')
    mkdir(out_dir_30m);
end

for doy = 1:TOTAL_DAYS

    fprintf('Export_X_30m | DOY %03d\n', doy);

    B_30m    = SR_B_match_30m{doy};
    G_30m    = SR_G_match_30m{doy};
    R_30m    = SR_R_match_30m{doy};
    NIR_30m  = SR_NIR_match_30m{doy};
    SWIR_30m = SR_SWIR_match_30m{doy};

    NDVI_daily_30m = NDVI_match_30m{doy};
    NDSI_daily_30m = NDSI_match_30m{doy};
    NDWI_daily_30m = NDWI_match_30m{doy};

    qeff_30m = qeff_match_30m{doy};
    snow_30m = snow_match_30m{doy};

    delta = 23.44 * sind(360/365 * (doy + 284));
    T = 3.63 + 104/15;
    H = 15 * (T - 12);

    cos_theta = sind(delta).*sind(lat_30m) + ...
                cosd(delta).*cosd(lat_30m).*cosd(H);
    cos_theta = min(max(cos_theta, -1), 1);

    theta = acosd(cos_theta);
    theta(theta > 90) = 90;

    sin_theta = sind(theta);
    sin_theta(sin_theta == 0) = eps;

    omiga = asind((cosd(delta).*sind(H))./sin_theta) + 180;

    cos_SAI = cosd(Slope_30m).*cosd(theta) + ...
              sind(Slope_30m).*sind(theta).*cosd(omiga - Aspect_30m);
    cos_SAI = min(max(cos_SAI, -1), 1);

    SAI_30m = acosd(cos_SAI);

    X_30m = [ ...
        B_30m(idx_30m), G_30m(idx_30m), R_30m(idx_30m), NIR_30m(idx_30m), SWIR_30m(idx_30m), ...
        NDVI_daily_30m(idx_30m), NDSI_daily_30m(idx_30m), NDWI_daily_30m(idx_30m), ...
        qeff_30m(idx_30m), snow_30m(idx_30m), ...
        Elv_30m(idx_30m), Slope_30m(idx_30m), Aspect_30m(idx_30m), SVF_30m(idx_30m), SSP_30m(idx_30m), ...
        lat_30m(idx_30m), SAI_30m(idx_30m), CLCD_30m(idx_30m) ...
    ];

    T_out_30m = array2table(X_30m, 'VariableNames', feature_names);
    csv_name_30m = sprintf('X_30m_%03d.csv', doy);
%     writetable(T_out_30m, fullfile(out_dir_30m, csv_name_30m));

end