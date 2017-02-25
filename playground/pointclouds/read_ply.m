files = dir()

ptCloud = pcread('helix_noface.ply');
figure;
pcshow(ptCloud);

%{
for file = 1 : size(files)
    if (contains(files(file).name, '.ply') | contains(files(file).name, '.pcd'))
        ptCloud = pcread(files(file).name);
        figure;
        pcshow(ptCloud);
    end
end

ptCloud = pcread('test.pcd');
figure;
pcshow(ptCloud);
%}