%% MODIS LST ATC Fitting (1 km)
%
% Author        : Zhao Junli
% Last Modified : 2026-01-28
%
% Description:
% This script performs pixel-wise ATC fitting for MODIS land surface
% temperature (LST) using NDVI as the vegetation constraint.
%
% Notes:
% 1) The ATC model is fitted using MOD11A1 daily LST and MOD13A2 NDVI products.
% 2) Since MOD13A2 is not available at daily resolution, NDVI time series are
%   temporally interpolated and expanded to daily scale to match the LST DOY.
% 3) LST and NDVI filenames are expected to contain date tokens in format:
%   YYYY_MM_DD (e.g., 2022_01_01.tif)

clc;
clear;
warning off


%% PART 0. USER-DEFINED PARAMETERS

EPSG_CODE   = 32648;           % Coordinate reference system (UTM Zone 48N)
TOTAL_DAYS  = 365;     
REF_YEAR    = 2022;   

% INPUT PATHS
InputPath_LST  = 'H:\WL_30m_LST\Data\LST_1km\';
InputPath_NDVI = 'H:\WL_30m_LST\Data\MOD13A2\';

% OUTPUT PATHS
OutputPath_ATC      = 'H:\WL_30m_LST_V2\Process\ATC\';
OutputPath_Residual = 'H:\WL_30m_LST_V2\Process\LST_residual_1km\';


% Create output directories if not exist
if ~exist(OutputPath_ATC, 'dir'); mkdir(OutputPath_ATC); end
if ~exist(OutputPath_Residual, 'dir'); mkdir(OutputPath_Residual); end


%% PART 1. READ MOD11 LST AND APPLY QC FILTERING

LST_files = dir(fullfile(InputPath_LST,'*.tif'));
nLST      = numel(LST_files);

LST_dates = datetime.empty(nLST,0);
LST_doy   = zeros(nLST,1); 

for t = 1:nLST

    fprintf('LST | %d / %d\n', t, nLST);

    fname = LST_files(t).name;

    date_token = regexp(fname,'(\d{4})_(\d{2})_(\d{2})','tokens','once');
    LST_dates(t) = datetime( ...
        str2double(date_token{1}), ...
        str2double(date_token{2}), ...
        str2double(date_token{3}) );

    LST_doy(t) = day(LST_dates(t),'dayofyear');

    % Read LST GeoTIFF
    [LST_raw, R_tmp] = readgeoraster(fullfile(InputPath_LST,fname));
    LST_raw = double(LST_raw);

    % Band definition
    LST_val = LST_raw(:,:,1) * 0.02;   % LST (K)
    LST_qa  = LST_raw(:,:,2);          % QC

    if t == 1
        [nRow,nCol] = size(LST_val);
        LST_ts = nan(nRow,nCol,nLST);
        R_LST  = R_tmp;
    end

    % QA mask
    mask_valid = ~isnan(LST_val);
    LST_tmp    = LST_val;

    if any(mask_valid(:))

        qc     = LST_qa(mask_valid);
        qc_bin = dec2bin(qc,8);

        qc1 = bin2dec(qc_bin(:,7:8));   % mandatory QA
        qc3 = bin2dec(qc_bin(:,3:4));   % emissivity error
        qc4 = bin2dec(qc_bin(:,1:2));   % LST error

        good = (qc1 == 0) | ...
               (qc1 == 1 & qc3 < 2 & qc4 < 2);

        mask_good = false(nRow,nCol);
        mask_good(mask_valid) = good;

        LST_tmp(~mask_good) = NaN;

    else
        LST_tmp(:) = NaN;
    end

    LST_ts(:,:,t) = LST_tmp;
end


%% PART 2. READ MOD13 NDVI AND APPLY QA FILTERING

NDVI_files = dir(fullfile(InputPath_NDVI,'*.tif'));
nNDVI      = numel(NDVI_files);

NDVI_dates = datetime.empty(nNDVI,0);
NDVI_doy   = zeros(nNDVI,1);

for t = 1:nNDVI

    fprintf('NDVI | %d / %d\n', t, nNDVI);

    fname = NDVI_files(t).name;

    date_token = regexp(fname,'(\d{4})_(\d{2})_(\d{2})','tokens','once');
    NDVI_dates(t) = datetime( ...
        str2double(date_token{1}), ...
        str2double(date_token{2}), ...
        str2double(date_token{3}) );

    NDVI_doy(t) = days(NDVI_dates(t) - datetime(REF_YEAR,1,1)) + 1;

    [NDVI_raw, ~] = readgeoraster(fullfile(InputPath_NDVI, fname));
    NDVI_raw = double(NDVI_raw);

    NDVI_val   = NDVI_raw(:,:,1)  * 0.0001;   
    DetailedQA = NDVI_raw(:,:,3);
    SummaryQA  = NDVI_raw(:,:,12);

    if t == 1
        [nRow,nCol] = size(NDVI_val);
        NDVI_ts = nan(nRow,nCol,nNDVI);
    end

    % QA mask
    mask_valid = ~isnan(NDVI_val);
    NDVI_tmp   = NDVI_val;

    if any(mask_valid(:))

        qa     = DetailedQA(mask_valid);
        qa_bin = dec2bin(qa,16);

        vi_quality = bin2dec(qa_bin(:,15:16));   % bit 0–1
        vi_use     = bin2dec(qa_bin(:,11:14));   % bit 2–5

        sumQA = SummaryQA(mask_valid);

        good = (sumQA == 0) | ...
               (sumQA == 1 & vi_quality <= 1 & vi_use <= 1);

        mask_good = false(nRow,nCol);
        mask_good(mask_valid) = good;

        NDVI_tmp(~mask_good) = NaN;

    else
        NDVI_tmp(:) = NaN;
    end

    NDVI_ts(:,:,t) = NDVI_tmp;
end


%% PART 3. NDVI TEMPORAL INTERPOLATION TO DAILY SCALE (MATCH LST DOY)

[nRow, nCol, ~] = size(NDVI_ts);
nLST = numel(LST_doy);

NDVI_daily = nan(nRow, nCol, nLST);

for r = 1:nRow
    fprintf('NDVI interp | row %d / %d\n', r, nRow);

    for c = 1:nCol

        ts_ndvi = squeeze(NDVI_ts(r, c, :));

        if all(isnan(ts_ndvi))
            continue;
        end

        valid = ~isnan(ts_ndvi);
        if sum(valid) < 4
            continue;
        end

        doy_v  = NDVI_doy(valid);
        ndvi_v = ts_ndvi(valid);

        ndvi_interp = interp1(doy_v, ndvi_v, LST_doy, 'linear');

        % Boundary fill
        ndvi_interp(LST_doy < min(doy_v)) = ndvi_v(1);
        ndvi_interp(LST_doy > max(doy_v)) = ndvi_v(end);

        NDVI_daily(r,c,:) = ndvi_interp;
    end
end


%% PART 4. PIXEL-WISE ATC FITTING (CORE LOGIC UNCHANGED)

[rows, cols, nLST] = size(LST_ts);

Alpha_map = NaN(rows, cols);
Beta_map  = NaN(rows, cols);
RMSE_map  = NaN(rows, cols);
R2_map    = NaN(rows, cols);
Num_map   = NaN(rows, cols);

Residuals_cube = NaN(rows, cols, nLST);

% ATC model 
atc_func = @(b, X) ...
    b(1) + (b(2) ./ (1 + exp(X(:,1)))) .* ...
    sin((2*pi .* (X(:,2) - 79)) ./ X(:,3));

for r = 1:rows
    fprintf('ATC fitting | row %d / %d\n', r, rows);

    for c = 1:cols

        Y_LST  = squeeze(LST_ts(r,c,:));
        X_NDVI = squeeze(NDVI_daily(r,c,:));

        valid = ~isnan(Y_LST) & ~isnan(X_NDVI);

        if sum(valid) < 35
            continue;
        end

        y_fit   = Y_LST(valid);
        x1_ndvi = X_NDVI(valid);
        x2_doy  = LST_doy(valid);
        x3_N    = TOTAL_DAYS * ones(sum(valid),1);

        X_mat = [x1_ndvi, x2_doy, x3_N];

        % Initial guess
        beta0 = [mean(y_fit), std(y_fit)];

        try
            [beta, ~, ~, ~, MSE] = nlinfit(X_mat, y_fit, atc_func, beta0);

            Alpha_map(r,c) = beta(1);
            Beta_map(r,c)  = beta(2);
            RMSE_map(r,c)  = sqrt(MSE);
            Num_map(r,c)   = sum(valid);

            y_pred = atc_func(beta, X_mat);
            R2_map(r,c) = 1 - sum((y_fit - y_pred).^2) / ...
                               sum((y_fit - mean(y_fit)).^2);

            % Residuals (unchanged)
            res_tmp = NaN(nLST,1);
            res_tmp(valid) = y_fit - y_pred;
            Residuals_cube(r,c,:) = res_tmp;

        catch
            continue;
        end
    end
end


%% PART 5. EXPORT RESULTS (GeoTIFF)

% Parameter maps
geotiffwrite(fullfile(OutputPath_ATC, sprintf('Alpha_1km_%d.tif', REF_YEAR)), ...
    Alpha_map, R_LST, 'CoordRefSysCode', EPSG_CODE);

geotiffwrite(fullfile(OutputPath_ATC, sprintf('Beta_1km_%d.tif', REF_YEAR)), ...
    Beta_map, R_LST, 'CoordRefSysCode', EPSG_CODE);

geotiffwrite(fullfile(OutputPath_ATC, sprintf('RMSE_1km_%d.tif', REF_YEAR)), ...
    RMSE_map, R_LST, 'CoordRefSysCode', EPSG_CODE);

geotiffwrite(fullfile(OutputPath_ATC, sprintf('R2_1km_%d.tif', REF_YEAR)), ...
    R2_map, R_LST, 'CoordRefSysCode', EPSG_CODE);

geotiffwrite(fullfile(OutputPath_ATC, sprintf('Num_1km_%d.tif', REF_YEAR)), ...
    Num_map, R_LST, 'CoordRefSysCode', EPSG_CODE);

% Residual layers by DOY
for k = 1:nLST
    fprintf('Residual export | %d / %d\n', k, nLST);

    res_layer = squeeze(Residuals_cube(:,:,k));
    doy_val   = LST_doy(k);

    filename = sprintf('residuals_%03d.tif', doy_val);

    geotiffwrite(fullfile(OutputPath_Residual, filename), ...
        res_layer, R_LST, 'CoordRefSysCode', EPSG_CODE);
end
