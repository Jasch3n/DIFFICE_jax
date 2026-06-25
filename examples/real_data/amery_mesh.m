md = bamg(model, 'domain', 'DomainOutline.exp');

bedmch_info = ncinfo('/Users/jiapchen/Research/Data/BedMachineAntarctica-v4.nc');

bedmch_x = ncread('/Users/jiapchen/Research/Data/BedMachineAntarctica-v4.nc', 'x')';
bedmch_y = ncread('/Users/jiapchen/Research/Data/BedMachineAntarctica-v4.nc', 'y')';
% bedmch_y = fliplr(bedmch_y);
bedmch_mask = ncread('/Users/jiapchen/Research/Data/BedMachineAntarctica-v4.nc', 'mask')';

%%
translation_vec = [-2.55e6, -2.15e6];
md_x = md.mesh.x+translation_vec(1);
md_y = md.mesh.y+translation_vec(2);
figure; hold on;

imagesc(bedmch_x/1e3, bedmch_y/1e3, bedmch_mask);

% plotmodel(md, 'figure', 1, 'data', 'mesh');
scatter(md_x/1e3, md_y/1e3, 0.25, 'filled', 'markerfacecolor', 'r', 'alphadata', 0.5);

axis equal;

xlim([min(md_x)/1e3-50, max(md_x)/1e3+50]);
ylim([min(md_y)/1e3-50, max(md_y)/1e3+50]);

xlabel('x_{ps} [km]');
ylabel('y_{ps} [km]')