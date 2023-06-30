% Created on Thu Apr 27 2023
% 
% author: Daniel Vázquez Pombo
% email: daniel.vazquez.pombo@gmail.com
% LinkedIn: https://www.linkedin.com/in/dvp/
% ResearchGate: https://www.researchgate.net/profile/Daniel-Vazquez-Pombo
% 
% I got reached out by several people complaining about SOLETE not being
% compatible with MATLAB and asking me to share the dataset in csv. That
% would be s̶̶t̶u̶p̶i̶d impractical given the dataset size, loading speed, etc.
%
% Nevertheless, here is a script allowing you to import SOLETE as close as
% possible to the Python version.
%
% Have fun!
%
% If you encounter any troubles using it let me know and I will see what I can do.
% 
% The licensing of this work is pretty chill, just give credit: https://creativecommons.org/licenses/by/4.0/
% 
%% Select File to Import
clc, clear,
FILE = 'SOLETE_Pombo_60min.h5'; % str - the filename is enough if you save this 
% file in the same directory as SOLETE, otherwise input the full directory
%options: SOLETE_short.h5 - SOLETE_Pombo_1sec.h5 - SOLETE_Pombo_1min.h5 -
% - SOLETE_Pombo_5min.h5 - SOLETE_Pombo_60min.h5

%% Miscelanea of hdf5-related functions
%h5disp(FILE) % use this to print the internal structure of the hdf5 file
% which is only important if you want to understand how MATLAB loads SOLETE
%There you can see that the Group is called DATA which has then 4 Datasets:
% axis0, axis1, block0_items and block0_values 

%now you can inport using h5 while pointing to the specific dataset
% h5read(FILE, '/DATA/axis0'); %columns of the original dataframe, names of the variables
% h5read(FILE, '/DATA/axis1'); %TIMESTAMP, index of the original dataframe
% h5read(FILE, '/DATA/block0_items'); %columns of the original dataframe, names of the variables
% h5read(FILE, '/DATA/block0_values')'; %values contained in the dataframe, column order matches the original dataset
%for some reason, MATLAB transposes the matrix, so we need to transpose it again to get the original shape

%% Import SOLETE

%this should build a table with the same column order as the original Python DataFrame
%however, for whatever reason, MATLAB decides to round values in a very stupid way
%if you find a way to solve that please let me know
% SOLETE = array2table([h5read(FILE, '/DATA/axis1'),h5read(FILE, '/DATA/block0_values')'], ...
%     'VariableNames', ['TIMESTAMP', h5read(FILE, '/DATA/block0_items')'])

%Therefore we must import it in this way. The only difference with the original dataset 
% is that the TIMESTAMP column is now the last one instead of the first one
SOLETE = array2table(h5read(FILE, '/DATA/block0_values')', 'VariableNames', ...
    h5read(FILE, '/DATA/block0_items'));
SOLETE.TIMESTAMP = h5read(FILE, '/DATA/axis1');

SOLETE = sortrows(SOLETE,size(SOLETE,2)); %This sorts the matrix based on timestamp
%In some cases Matlab messes with SOLETE's order.

SOLETE.Datetime = datetime(SOLETE.Timestamp./1e9,'ConvertFrom','posixtime',...
                                            'Format', 'yyyy-MM-dd HH:mm:ss');
%this last function is optional, you only need it if you want to convert 
%the timestamps into formated datetime values

%Lines 55 and 58 were submitted by Jon Martinez-Rico.
%We all heil Jon and bow to his neverending knowledge.

%Note that the TIMESTAMPS are in epoch time. The first one should be:
% -> GMT: Friday, June 1, 2018 00:00:00

disp('SOLETE has been imported');
