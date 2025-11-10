function process_final_data()
    % 定义晶体阵列的尺寸（如需可修改）
    NUM_ROWS = 44;
    NUM_COLS = 44;

    % 基本路径
    base_dir = pwd; % 当前工作目录（c:\Users\Erinneria\Downloads\数据）
    tasks_dir = fullfile(base_dir, 'tasks');
    results_dir = fullfile(base_dir, 'results');

    % 1. 加载全局必要文件（classificationmap, calibration_params）
    fprintf('加载全局标定与分类文件...\n');
    try
        map_data = load(fullfile(base_dir, 'classificationmap.mat'));
        classificationmap = map_data.classificationmap;
        params_data = load(fullfile(base_dir, 'calibration_params.mat'));
        calibration_params = params_data.calibration_params;
    catch ME
        error('加载 classificationmap.mat 或 calibration_params.mat 失败: %s', ME.message);
    end

    % 检查 tasks 目录是否存在
    if ~isfolder(tasks_dir)
        error('未找到 tasks 目录: %s', tasks_dir);
    end

    % 遍历 tasks 目录下的每个子文件夹
    task_entries = dir(tasks_dir);
    task_entries = task_entries([task_entries.isdir]); % 仅目录
    % 过滤 . 和 ..
    task_entries = task_entries(~ismember({task_entries.name}, {'.', '..'}));

    for ti = 1:numel(task_entries)
        task_name = task_entries(ti).name;
        task_path = fullfile(tasks_dir, task_name);
        data_folder = fullfile(task_path, 'data');
        h5_file = fullfile(data_folder, 'original_data.h5');

        fprintf('\n处理任务: %s\n', task_name);

        if ~isfile(h5_file)
            fprintf('  跳过：未找到文件 %s\n', h5_file);
            continue;
        end

        % 尝试优先读取 /daq0/integral 与 /daq0/timestamps
        try
            use_preferred = false;
            try
                integral = h5read(h5_file, '/daq0/integral');
                timestamps = h5read(h5_file, '/daq0/timestamps');
                % 转为 double
                integral = double(integral);
                timestamps = double(timestamps);

                % 规范 integral 为 (N_events x nChannels) 形式 —— 根据 timestamps 长度判断是否需要转置
                nT = numel(timestamps);
                sz = size(integral);
                fprintf('  /daq0/integral 原始形状: [%d %d], timestamps 长度: %d\n', sz(1), sz(2), nT);

                if nT > 1
                    if sz(1) == nT
                        % already N x C
                    elseif sz(2) == nT
                        integral = integral.'; % 转为 N x C
                        fprintf('  已转置 integral 为 [%d %d]\n', size(integral,1), size(integral,2));
                    else
                        % 如果两者都不匹配，使用启发式：较大的维度视为事件维度
                        if sz(1) < sz(2)
                            integral = integral.'; % 更可能是 C x N -> 转置
                            fprintf('  使用启发式转置 integral 为 [%d %d]\n', size(integral,1), size(integral,2));
                        else
                            fprintf('  启发式：保持 integral 形状 [%d %d]\n', sz(1), sz(2));
                        end
                    end
                else
                    % timestamps 长度为1 或不可用，按常规把较小维度视为通道维度
                    if sz(1) < sz(2)
                        integral = integral.'; 
                        fprintf('  timestamps 单值，已转置 integral 为 [%d %d]\n', size(integral,1), size(integral,2));
                    end
                end

                % 现在 integral 应为 N x C
                % 检查列数（通道数）
                if size(integral,2) < 4
                    error('daq0/integral 列数不足（需要 >=4 列），当前通道数: %d', size(integral,2));
                end

                % 保证 timestamps 长度与 integral 行数一致，取最小长度
                nEvents = min(size(integral,1), numel(timestamps));
                integral = integral(1:nEvents,1:4);
                timestamps = timestamps(1:nEvents);

                % 组合为 raw_data: 第一列 timestamps，其后为 4 列 ADC
                raw_data = [timestamps(:), integral(:,1:4)];
                use_preferred = true;
                fprintf('  使用 /daq0/integral 和 /daq0/timestamps（事件: %d, 通道: %d）\n', nEvents, size(integral,2));
            catch prefME
                % 如果优先路径失败，回退到自动查找第一个数值数据集
                fprintf('  未能使用 /daq0/integral 或 /daq0/timestamps (%s)，回退自动检测。\n', prefME.message);
            end

            if ~use_preferred
                info = h5info(h5_file);
                ds_path = find_first_numeric_dataset(info, '/');
                if isempty(ds_path)
                    fprintf('  跳过：在 %s 中未找到数值数据集。\n', h5_file);
                    continue;
                end

                raw_data = h5read(h5_file, ds_path);
                % 转为 double 且确保为二维数组（事件 x 列）
                raw_data = double(raw_data);
                if isvector(raw_data)
                    raw_data = raw_data(:); % 列向量
                end
                % 如果数据集维度为 (channels, events) 而非 (events, channels)，尝试转置
                if size(raw_data,1) < size(raw_data,2) && (size(raw_data,1) <= 4) && (size(raw_data,2) >= 5)
                    raw_data = raw_data.'; % 转置为 events x channels
                end
            end

        catch ME
            fprintf('  读取 HDF5 失败: %s\n', ME.message);
            continue;
        end

        % 验证数据列数足够（脚本期望至少有 5 列，ADC 在列 2:5）
        if size(raw_data,2) < 5
            fprintf('  跳过：数据列不足 (需要 >=5 列)，当前列数: %d\n', size(raw_data,2));
            continue;
        end

        % === 原处理流程（向量化） ===
        event_data = raw_data;
        fprintf('  读取事件数: %d，开始计算位置与标定能量...\n', size(event_data,1));
        
        % 读取第一列的时间戳
        timestamps = event_data(:, 1);
        datas = event_data(:, 2:5);
        adc_sums = sum(datas, 2);

        % 避免除以零
        adc_sums_nozero = adc_sums;
        adc_sums_nozero(adc_sums_nozero == 0) = eps;

        x_coords = round((datas(:,1) + datas(:,2) - datas(:,3) - datas(:,4)) ./ adc_sums_nozero * 200) + 256;
        y_coords = round((datas(:,1) - datas(:,2) - datas(:,3) + datas(:,4)) ./ adc_sums_nozero * 200) + 256;

        num_events = size(event_data, 1);
        final_results = nan(num_events, 4);

        valid_pos_mask = (x_coords >= 1 & x_coords <= 512 & y_coords >= 1 & y_coords <= 512);

        valid_x = x_coords(valid_pos_mask);
        valid_y = y_coords(valid_pos_mask);
        valid_adc_sums = adc_sums(valid_pos_mask);

        linear_indices = sub2ind(size(classificationmap), valid_y, valid_x);
        crystal_ids = classificationmap(linear_indices);

        % 校验 crystal_ids 在 calibration_params 范围
        valid_id_mask = crystal_ids >= 1 & crystal_ids <= numel(calibration_params);
        if any(~valid_id_mask)
            % 对于超范围的 ID，置为 NaN（丢弃）
            warning('  部分晶体ID超出 calibration_params 范围 (%d 个)。这些事件将被置空。', sum(~valid_id_mask));
            crystal_ids(~valid_id_mask) = 1; % 临时索引以避免下标越界，后面会将能量设为 NaN
        end

        k_values = calibration_params(crystal_ids);
        % 对被标记为无效 id 的事件，设置 k 为 NaN
        if any(~valid_id_mask)
            k_values(~valid_id_mask) = NaN;
        end

        calibrated_energies = valid_adc_sums .* k_values;

        final_results(:, 1) = timestamps;
        final_results(valid_pos_mask, 2) = valid_x;
        final_results(valid_pos_mask, 3) = valid_y;
        final_results(valid_pos_mask, 4) = calibrated_energies;

        % === 保存结果到 results/<task>/data/position.npy ===
        out_data_folder = fullfile(results_dir, task_name, 'data');
        if ~isfolder(out_data_folder)
            mkdir(out_data_folder);
        end
        out_file = fullfile(out_data_folder, 'position.npy');

        try
            % 使用 py.numpy.save 保存，确保路径为 char
            py.numpy.save(out_file, final_results);
            num_valid_events = sum(~isnan(final_results(:,3)));
            fprintf('  已保存：%s (事件: %d，有效: %d)\n', out_file, num_events, num_valid_events);
        catch ME
            fprintf('  保存到 %s 失败: %s\n', out_file, ME.message);
            continue;
        end
    end

    fprintf('\n全部任务处理完成。\n');
end

% 辅助函数：在 h5info 结构中递归查找第一个数值数据集路径（返回 HDF5 路径，如 /group/dataset）
function ds_path = find_first_numeric_dataset(h5info_struct, current_path)
    ds_path = '';
    % 检查当前结构下的数据集
    if isfield(h5info_struct, 'Datasets') && ~isempty(h5info_struct.Datasets)
        for ii = 1:numel(h5info_struct.Datasets)
            d = h5info_struct.Datasets(ii);
            % 仅选择数值类型（排除字符串等）
            if ismember(d.Datatype.Class, {'H5T_INTEGER', 'H5T_FLOAT'})
                ds_path = fullfile(current_path, d.Name);
                % MATLAB fullfile 用反斜杠，会导致路径不是 HDF5 标准路径，用 slash 修正
                ds_path = strrep(ds_path, '\', '/');
                return;
            end
        end
    end
    % 递归检查子组
    if isfield(h5info_struct, 'Groups') && ~isempty(h5info_struct.Groups)
        for gi = 1:numel(h5info_struct.Groups)
            g = h5info_struct.Groups(gi);
            subpath = fullfile(current_path, g.Name);
            subpath = strrep(subpath, '\', '/');
            ds_path = find_first_numeric_dataset(g, subpath);
            if ~isempty(ds_path)
                return;
            end
        end
    end
end