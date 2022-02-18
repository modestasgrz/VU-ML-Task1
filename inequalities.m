% Function defining 1st task's inequality system which is needed to find weights and bias

% Input - matrix of weights - columns define weight number, rows - different weights in search

function inequalities(theta)

	weight1 = theta;
	weight2 = theta;
	bias = theta;
	
	[weight1, weight2, bias] = meshgrid(theta);
	
	condition1 = bias - 0.3*weight1 + 0.6*weight2 < 0;
	condition2 = bias + 0.3*weight1 - 0.6*weight2 < 0;
	condition3 = bias + 1.2*weight1 - 1.2*weight2 > 0;
	condition4 = bias + 1.2*weight1 + 1.2*weight2 > 0;
	
	condition = condition1 & condition2 & condition3 & condition4;
	
	% colors = zeros(size(weight1)) + condition1 + condition2 + condition3 + condition4;
	% scatter3(weight1(:), weight2(:), bias(:), 3, colors(:), 'filled')
	
	scatter3(weight1(condition), weight2(condition), bias(condition), 'b', 'filled')

end