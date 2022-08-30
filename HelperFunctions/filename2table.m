function table_all = filename2table(fds,var1Name,lable,var2Name)
% Function to create the table include file names from fds.
name_table = table();
lables_table = table();

[fds_size,~] = size(fds.Files);
fds_path_name = fullfile(fds.Files);

for i = 1: fds_size   
    [~,filename,~] = fileparts(fds_path_name{i});
    name_table = [name_table; table({filename})];
    lables_table = [lables_table; table(lable)]; 
end

name_table.Properties.VariableNames = {var1Name};
lables_table.Properties.VariableNames = {var2Name};

table_all = [name_table lables_table];

end