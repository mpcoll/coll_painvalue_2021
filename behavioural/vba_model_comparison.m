% Load model evidence
table = readtable('model_comparison.csv');

AIC_models = [table.none_AIC, table.linear_AIC, table.expo_AIC,...
              table.para_AIC];

% Run VBA model comparison
options.modelNames = {'none', 'linear', 'expo', 'para'};
[posterior,out] = VBA_groupBMC(AIC_models'*-1, options) ;
% Calculate protected exceedance probability
out.pep = (1-out.bor)*out.ep + out.bor/length(out.ep);
% Save
save(fullfile(pwd, 'VBA_model_comp'), 'posterior', 'out')
