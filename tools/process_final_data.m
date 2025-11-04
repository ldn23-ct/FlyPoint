function process_final_data()
    % 定义晶体阵列的尺寸，必须与 biaoding.m 中的一致
    NUM_ROWS = 44;
    NUM_COLS = 44;

    base_dir = "C:\\Users\\46595\\PHDLearning\\backscatter\\背散射实验平台结构\\平台结构\\飞点\\code"
    task_dir = fullfile(base_dir, 'TrueData\\2025_10_29_15_22_13');
    % 1. 加载所有必要文件
    fprintf('1/4: 正在加载输入文件...\n');
    try
        event_data = importdata(fullfile(task_dir, 'Energy.txt'));
        event_data = event_data(:, [1, 3:6]);
        map_data = load('classificationmap.mat');
        classificationmap = map_data.classificationmap;
        params_data = load('calibration_params.mat');
        calibration_params = params_data.calibration_params;
        
        fprintf('    - 成功加载 Energy.txt (%d 个事件)。\n', size(event_data, 1));
        fprintf('    - 成功加载 classificationmap.mat。\n');
        fprintf('    - 成功加载 calibration_params.mat。\n');
        
    catch ME
        error('加载文件失败: %s。请确保 Energy.txt, classificationmap.mat, 和 calibration_params.mat 都在当前路径下。', ME.message);
    end

    % 2. 向量化计算 x, y 坐标和 ADC 总和
    fprintf('2/4: 正在计算所有事件的 x, y 坐标和 ADC 总和...\n');
    datas = event_data(:, 2:5);
    adc_sums = sum(datas, 2);

    % 为避免除以零，将 adc_sums 中的零替换为 eps (一个极小的正数)
    adc_sums_nozero = adc_sums;
    adc_sums_nozero(adc_sums_nozero == 0) = eps;
    
    x_coords = round((datas(:,1) + datas(:,2) - datas(:,3) - datas(:,4)) ./ adc_sums_nozero * 200) + 256;
    y_coords = round((datas(:,1) - datas(:,2) - datas(:,3) + datas(:,4)) ./ adc_sums_nozero * 200) + 256;

    % 3. 查找晶体ID，计算能量，并转换为行列号
    fprintf('3/4: 正在计算行列号和标定能量...\n');
    
    num_events = size(event_data, 1);
    % 初始化最终结果矩阵 [行号, 列号, 能量]，所有值默认为 NaN
    final_results = nan(num_events, 3);

    % 创建一个逻辑掩码，标记所有位置坐标在 [1, 512] 范围内的有效事件
    valid_pos_mask = (x_coords >= 1 & x_coords <= 512 & y_coords >= 1 & y_coords <= 512);
    
    % --- 只对位置有效的事件进行后续处理 ---
    
    % 获取这些有效事件的坐标和原始ADC和
    valid_x = x_coords(valid_pos_mask);
    valid_y = y_coords(valid_pos_mask);
    valid_adc_sums = adc_sums(valid_pos_mask);
    
    % 查找线性晶体ID (z)
    linear_indices = sub2ind(size(classificationmap), valid_y, valid_x);
    crystal_ids = classificationmap(linear_indices);
    
    % 使用晶体ID作为索引，获取每个有效事件对应的标定系数 k
    k_values = calibration_params(crystal_ids);
    
    % 计算标定后的能量 (e)
    calibrated_energies = valid_adc_sums .* k_values;
    
    % 将线性的 crystal_ids 转换为 (行, 列) 序号
    [crystal_rows, crystal_cols] = ind2sub([NUM_ROWS, NUM_COLS], crystal_ids);
    
    % 将计算出的 [行, 列, 能量] 填充回结果矩阵的相应位置
    final_results(valid_pos_mask, 1) = crystal_rows;
    final_results(valid_pos_mask, 2) = crystal_cols;
    final_results(valid_pos_mask, 3) = calibrated_energies;
    
    % 4. 保存最终结果到 .txt 文件
    fprintf('4/4: 正在将最终结果保存到 final_event_list.npy...\n');
    final_event_list = final_results;

    writeNPY(final_event_list, fullfile(task_dir,'final_event_list.npy'));
    
    % 显示统计信息
    num_valid_events = sum(~isnan(final_event_list(:,3)));
    fprintf('    - 保存完成！\n');
    fprintf('\n--- 统计摘要 ---\n');
    fprintf('总事件数: %d\n', num_events);
    fprintf('有效事件数 (有位置、有晶体ID、有标定能量): %d (%.2f%%)\n', num_valid_events, (num_valid_events/num_events)*100);
    fprintf('最终结果矩阵 (大小: %d x %d) 已保存到 position.txt。\n', size(final_event_list,1), size(final_event_list,2));
    fprintf('矩阵列定义: [行号, 列号, 能量(keV)]\n');
end
