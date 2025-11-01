--[[

	--------------------------------------------------------------------

	Aqwam's 2D Tensor Library (TensorL-2D)

	Version: 1.0

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
	
	By using or possesing any copies of this library, you agree to our Terms and Conditions at:
	
	https://github.com/AqwamCreates/TensorL-2D/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = {}

local function deepCopyTable(original, copies)

	copies = copies or {}

	local originalType = type(original)

	local copy

	if (originalType == 'table') then

		if copies[original] then

			copy = copies[original]

		else

			copy = {}

			copies[original] = copy

			for originalKey, originalValue in next, original, nil do

				copy[deepCopyTable(originalKey, copies)] = deepCopyTable(originalValue, copies)

			end

			setmetatable(copy, deepCopyTable(getmetatable(original), copies))

		end

	else -- number, string, boolean, etc

		copy = original

	end

	return copy

end

local function onBroadcastError(tensor1, tensor2)

	local errorMessage = "Unable To Broadcast. \n" .. "Tensor 1 Size: " .. "(" .. #tensor1 .. ", " .. #tensor1[1] .. ") \n" .. "Tensor 2 Size: " .. "(" .. #tensor2 .. ", " .. #tensor2[1] .. ") \n"

	error(errorMessage)

end

local function checkIfCanBroadcast(tensor1, tensor2)

	local tensor1Rows = #tensor1

	local tensor2Rows = #tensor2

	local tensor1Columns = #tensor1[1]

	local tensor2Columns = #tensor2[1]

	local isTensor1Broadcasted
	local isTensor2Broadcasted

	local hasSameRowSize = (tensor1Rows == tensor2Rows)

	local hasSameColumnSize = (tensor1Columns == tensor2Columns)

	local hasSameDimension = hasSameRowSize and hasSameColumnSize

	local isTensor1IsLargerInOneDimension = ((tensor1Rows > 1) and hasSameColumnSize and (tensor2Rows == 1)) or ((tensor1Columns > 1) and hasSameRowSize and (tensor2Columns == 1))

	local isTensor2IsLargerInOneDimension = ((tensor2Rows > 1) and hasSameColumnSize and (tensor1Rows == 1)) or ((tensor2Columns > 1) and hasSameRowSize and (tensor1Columns == 1))

	local isTensor1Scalar = (tensor1Rows == 1) and (tensor1Columns == 1)

	local isTensor2Scalar = (tensor2Rows == 1) and (tensor2Columns == 1)

	local isTensor1Larger = ((tensor1Rows > tensor2Rows) or (tensor1Columns > tensor2Columns)) and not ((tensor1Rows < tensor2Rows) or (tensor1Columns < tensor2Columns))

	local isTensor2Larger = ((tensor2Rows > tensor1Rows) or (tensor2Columns > tensor1Columns)) and not ((tensor2Rows < tensor1Rows) or (tensor2Columns < tensor1Columns))

	if (hasSameDimension) then

		isTensor1Broadcasted = false
		isTensor2Broadcasted = false

	elseif (isTensor2IsLargerInOneDimension) or (isTensor2Larger and isTensor1Scalar) then

		isTensor1Broadcasted = true
		isTensor2Broadcasted = false

	elseif (isTensor1IsLargerInOneDimension) or (isTensor1Larger and isTensor2Scalar) then

		isTensor1Broadcasted = false
		isTensor2Broadcasted = true

	else

		onBroadcastError(tensor1, tensor2)

	end

	return isTensor1Broadcasted, isTensor2Broadcasted

end

function AqwamTensorLibrary:expand(tensor, targetRowSize, targetColumnSize)

	local result = {}

	local isTensorRowSizeEqualToOne = (#tensor == 1)

	local isTensorColumnSizeEqualToOne = (#tensor[1] == 1)

	if (isTensorRowSizeEqualToOne) and (not isTensorColumnSizeEqualToOne) then

		for row = 1, targetRowSize, 1 do

			result[row] = {}

			for column = 1, targetColumnSize, 1 do result[row][column] = tensor[1][column] end

		end

	elseif (not isTensorRowSizeEqualToOne) and (isTensorColumnSizeEqualToOne) then

		for row = 1, targetRowSize, 1 do

			result[row] = {}

			for column = 1, targetColumnSize, 1 do result[row][column] = tensor[row][1] end

		end

	elseif (isTensorRowSizeEqualToOne) and (isTensorColumnSizeEqualToOne) then

		for row = 1, targetRowSize, 1 do

			result[row] = {}

			for column = 1, targetColumnSize, 1 do result[row][column] = tensor[1][1] end

		end

	end

	return result

end

local function broadcast(tensor1, tensor2, deepCopyOriginalTensor)

	local isTensor1Broadcasted, isTensor2Broadcasted = checkIfCanBroadcast(tensor1, tensor2)

	if (isTensor1Broadcasted) then tensor1 = AqwamTensorLibrary:expand(tensor1, #tensor2, #tensor2[1]) end

	if (isTensor2Broadcasted) then tensor2 = AqwamTensorLibrary:expand(tensor2, #tensor1, #tensor1[1]) end

	if (not isTensor1Broadcasted) and (deepCopyOriginalTensor) then tensor1 = deepCopyTable(tensor1) end

	if (not isTensor2Broadcasted) and (deepCopyOriginalTensor) then tensor2 = deepCopyTable(tensor2) end

	return tensor1, tensor2	

end

function AqwamTensorLibrary:broadcast(tensor1, tensor2)

	return broadcast(tensor1, tensor2, true)

end

local function convertToTensorIfScalar(value)

	local isNotScalar

	isNotScalar = pcall(function()

		local testForScalar = value[1][1]

	end)

	if not isNotScalar then

		return {{value}}

	else

		return value

	end

end

local function onDotProductError(tensor1Column, tensor2Row)

	local errorMessage = "Incompatible Tensor Dimensions: " .. tensor1Column .. " Column(s), " .. tensor2Row .. " Row(s)."

	error(errorMessage)

end

local function checkIfCanDotProduct(tensor1, tensor2)

	local tensor1Column = #tensor1[1]
	local tensor2Row = #tensor2

	if (tensor1Column ~= tensor2Row) then

		onDotProductError(tensor1Column, tensor2Row)

	end

end

local function dotProduct(tensor1, tensor2)

	local result = {}

	local tensor1Row = #tensor1
	local tensor1Column = #tensor1[1]
	local tensor2Column = #tensor2[1]
	
	local tensor1Array

	checkIfCanDotProduct(tensor1, tensor2)

	for row = 1, tensor1Row, 1 do
		
		local resultArray = {}
		
		tensor1Array = tensor1[row]

		for column = 1, tensor2Column, 1 do

			local sum = 0

			for i = 1, tensor1Column do sum = sum + (tensor1Array[i] * tensor2[i][column]) end

			resultArray[column] = sum

		end
		
		result[row] = resultArray

	end

	return result

end

local function generateArgumentErrorString(tensors, firstTensorIndex, secondTensorIndex)

	local text1 = "Argument " .. firstTensorIndex .. " and " .. secondTensorIndex .. " are incompatible! "

	local text2 = "(" ..  #tensors[firstTensorIndex] .. ", " .. #tensors[firstTensorIndex][1] .. ") and " .. "(" ..  #tensors[secondTensorIndex] .. ", " .. #tensors[secondTensorIndex][1] .. ")"

	local text = text1 .. text2

	return text

end

local function applyFunctionUsingOneTensor(functionToApply, tensor)

	local result = {}

	local resultrowVector

	for row, rowVector in ipairs(tensor) do

		resultrowVector = {}

		for column, value in ipairs(rowVector) do

			resultrowVector[column] = functionToApply(value)

		end

		result[row] = resultrowVector

	end

	return result

end

local function applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2)

	if (#tensor1 ~= #tensor2) or (#tensor1[1] ~= #tensor2[1]) then error("Incompatible Dimensions! (" .. #tensor1 .." x " .. #tensor1[1] .. ") and (" .. #tensor2 .. " x " .. #tensor2[1] .. ")") end

	local result = {}

	local resultrowVector
	
	local rowVector2

	for row, rowVector1 in ipairs(tensor1) do
		
		rowVector2 = tensor2[row]

		resultrowVector = {}
		
		for column, value in ipairs(rowVector1) do

			resultrowVector[column] = functionToApply(value, rowVector2[column])

		end

		result[row] = resultrowVector

	end

	return result

end

local function applyFunctionWhenTheFirstValueIsAScalar(functionToApply, scalar, tensor)

	local result = {}
	
	local resultrowVector
	
	for row, rowVector in ipairs(tensor) do
		
		resultrowVector = {}
		
		for column, value in ipairs(rowVector) do
			
			resultrowVector[column] = functionToApply(scalar, value)
			
		end
		
		result[row] = resultrowVector
		
	end

	return result

end

local function applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, scalar)

	local result = {}

	local resultrowVector

	for row, rowVector in ipairs(tensor) do

		resultrowVector = {}

		for column, value in ipairs(rowVector) do

			resultrowVector[column] = functionToApply(value, scalar)

		end

		result[row] = resultrowVector

	end

	return result

end

local function applyFunctionUsingMultipleTensors(functionToApply, ...)

	local tensors = {...}

	local numberOfTensors = #tensors

	local tensor = tensors[1]

	if (numberOfTensors == 1) then 

		if (type(tensor) == "table") then

			return applyFunctionUsingOneTensor(functionToApply, tensor) 

		else

			return functionToApply(tensor)

		end

	end

	for i = 2, numberOfTensors, 1 do

		local otherTensor = tensors[i]

		local isFirstValueIsTensor = (type(tensor) == "table")

		local isSecondValueIsTensor = (type(otherTensor) == "table")

		if (isFirstValueIsTensor) and (isSecondValueIsTensor) then

			tensor, otherTensor = broadcast(tensor, otherTensor, false)

			tensor = applyFunctionUsingTwoTensors(functionToApply, tensor, otherTensor)

		elseif (not isFirstValueIsTensor) and (isSecondValueIsTensor) then

			tensor = applyFunctionWhenTheFirstValueIsAScalar(functionToApply, tensor, otherTensor)

		elseif (isFirstValueIsTensor) and (not isSecondValueIsTensor) then

			tensor = applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, otherTensor)

		else

			tensor = functionToApply(tensor, otherTensor)

		end

	end

	return tensor

end

function AqwamTensorLibrary:unaryMinus(...)

	return applyFunctionUsingMultipleTensors(function(a) return -a end, ...)

end

function AqwamTensorLibrary:add(...)

	return applyFunctionUsingMultipleTensors(function(a, b) return a + b end, ...)

end

function AqwamTensorLibrary:subtract(...)

	return applyFunctionUsingMultipleTensors(function(a, b) return a - b end, ...)

end

function AqwamTensorLibrary:multiply(...)

	return applyFunctionUsingMultipleTensors(function(a, b) return a * b end, ...)

end

function AqwamTensorLibrary:divide(...)

	return applyFunctionUsingMultipleTensors(function(a, b) return a / b end, ...)

end

function AqwamTensorLibrary:logarithm(...)

	return applyFunctionUsingMultipleTensors(math.log, ...)

end

function AqwamTensorLibrary:exponent(...)

	return applyFunctionUsingMultipleTensors(math.exp, ...)

end

function AqwamTensorLibrary:power(...)

	return applyFunctionUsingMultipleTensors(math.pow, ...)

end

function AqwamTensorLibrary:areValuesEqual(...)

	return applyFunctionUsingMultipleTensors(function(a, b) return a == b end, ...)

end

function AqwamTensorLibrary:areValuesGreater(...)

	return applyFunctionUsingMultipleTensors(function(a, b) return a > b end, ...)

end

function AqwamTensorLibrary:areValuesGreaterOrEqual(...)

	return applyFunctionUsingMultipleTensors(function(a, b) return a >= b end, ...)

end

function AqwamTensorLibrary:areValuesLesser(...)

	return applyFunctionUsingMultipleTensors(function(a, b) return a < b end, ...)

end

function AqwamTensorLibrary:areValuesLesserOrEqual(...)

	return applyFunctionUsingMultipleTensors(function(a, b) return a <= b end, ...)

end

function AqwamTensorLibrary:areTensorsEqual(...)

	local resultTensor = applyFunctionUsingMultipleTensors(function(a, b) return a == b end, ...)
	
	for _, rowVector in ipairs(resultTensor) do
		
		for _, value in ipairs(rowVector) do
			
			if (not value) then return false end
			
		end
		
	end

	return true

end

function AqwamTensorLibrary:dotProduct(...)

	local tensors = {...}

	local result = tensors[1]
	
	local secondTensor

	result = convertToTensorIfScalar(result)

	for i = 2, #tensors, 1 do

		result = convertToTensorIfScalar(result)
		
		secondTensor = convertToTensorIfScalar(tensors[i])

		result = dotProduct(result, secondTensor)

	end

	return result

end

function AqwamTensorLibrary:createIdentityTensor(dimensionSizeArray, value)

	if (#dimensionSizeArray ~= 2) then error("Invalid dimension size array.") end

	local numberOfRows = dimensionSizeArray[1]

	local numberOfColumns = dimensionSizeArray[2]

	local result = {}
	
	local array
	
	value = value or 1

	for row = 1, numberOfRows, 1 do
		
		array = table.create(numberOfColumns, 0) 
		
		array[row] = value
		
		result[row] = array
		
	end

	return result

end

function AqwamTensorLibrary:createTensor(dimensionSizeArray, allValues)

	local numberOfRows = dimensionSizeArray[1]

	local numberOfColumns = dimensionSizeArray[2]
	
	allValues = allValues or 0

	local result = {}
	
	for row = 1, numberOfRows, 1 do result[row] = table.create(numberOfColumns, allValues) end

	return result

end

function AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)

	local numberOfRows = dimensionSizeArray[1]

	local numberOfColumns = dimensionSizeArray[2]

	local result = {}

	local random = Random.new()

	mean = mean or 0

	standardDeviation = standardDeviation or 1

	for row = 1, numberOfRows do

		result[row] = {}

		for column = 1, numberOfColumns do

			local randomNumber1 = random:NextNumber()

			local randomNumber2 = random:NextNumber()

			local zScore = math.sqrt(-2 * math.log(randomNumber1)) * math.cos(2 * math.pi * randomNumber2)

			result[row][column] = (zScore * standardDeviation) + mean

		end
	end

	return result

end

function AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue)

	local numberOfRows = dimensionSizeArray[1]

	local numberOfColumns = dimensionSizeArray[2]

	local result = {}
	
	if (minimumValue) and (maximumValue) then
		
		if (minimumValue >= maximumValue) then error("The minimum value cannot exceed the maximum value.") end
		
		local rangeValue = maximumValue - minimumValue
		
		for row = 1, numberOfRows, 1 do

			result[row] = {}

			for column = 1, numberOfColumns, 1 do

				result[row][column] = minimumValue + (math.random() * rangeValue)

			end

		end
		
	elseif (not minimumValue) and (maximumValue) then
		
		if (maximumValue <= 0) then error("The maximum value cannot be less than or equal to zero.") end

		for row = 1, numberOfRows, 1 do

			result[row] = {}

			for column = 1, numberOfColumns, 1 do

				result[row][column] = math.random() * maximumValue

			end	

		end

	elseif (minimumValue) and (not maximumValue) then
		
		if (minimumValue >= 0) then error("The minimum value cannot be greater than or equal to zero.") end

		for row = 1, numberOfRows, 1 do

			result[row] = {}

			for column = 1, numberOfColumns, 1 do

				result[row][column] = math.random() * minimumValue

			end	

		end
		
	elseif (not minimumValue) and (not maximumValue) then

		for row = 1, numberOfRows, 1 do

			result[row] = {}

			for column = 1, numberOfColumns, 1 do

				result[row][column] = (math.random() * 2) - 1

			end	

		end

	end

	return result

end

function AqwamTensorLibrary:getDimensionSizeArray(...)

	local tensorSizeArray = {}

	for i, tensor in ipairs({...}) do

		local numberOfRows = #tensor

		local numberOfColumns = #tensor[1]

		local dimensionSizeArray = {numberOfRows, numberOfColumns}

		table.insert(tensorSizeArray, dimensionSizeArray)

	end

	return table.unpack(tensorSizeArray)

end


function AqwamTensorLibrary:transpose(tensor)

	local numberOfRows = #tensor
	local numberOfColumns = #tensor[1]

	local result = AqwamTensorLibrary:createTensor({numberOfColumns, numberOfRows})
	
	for row, rowVector in ipairs(tensor) do

		for column, value in ipairs(rowVector) do

			result[column][row] = value

		end

	end

	return result

end

local function sumFromAllDimensions(tensor)

	local result = 0
	
	for _, rowVector in ipairs(tensor) do

		for _, value in ipairs(rowVector) do

			result = result + value

		end

	end

	return result

end

local function rowSum(tensor)

	local numberOfColumns = #tensor[1]

	local result = AqwamTensorLibrary:createTensor({1, numberOfColumns})
	
	for _, rowVector in ipairs(tensor) do
		
		for column, value in ipairs(rowVector) do
			
			result[1][column] = result[1][column] + value
			
		end
		
	end

	return result

end

local function columnSum(tensor)

	local numberOfRows = #tensor

	local result = AqwamTensorLibrary:createTensor({numberOfRows, 1})

	for row, rowVector in ipairs(tensor) do

		for _, value in ipairs(rowVector) do

			result[row][1] = result[row][1] + value

		end

	end

	return result

end

function AqwamTensorLibrary:sum(tensor, dimension)
	
	if (type(tensor) == "number") then return tensor end

	if (not dimension) then 

		return sumFromAllDimensions(tensor) 

	elseif (dimension == 1) then

		return rowSum(tensor)

	elseif (dimension == 2) then

		return columnSum(tensor)

	else

		error("Invalid dimension.")

	end

end

local function calculateMean(tensor)

	local sum = 0

	local numberOfElements = #tensor * #tensor[1]

	for _, unwrappedRowVector in ipairs(tensor) do
		
		for _, value in ipairs(unwrappedRowVector) do
			
			sum = sum + value
			
		end
		
	end

	local mean = sum / numberOfElements

	return mean

end

function AqwamTensorLibrary:mean(tensor, dimension)
	
	if (type(tensor) == "number") then return tensor end

	if (not dimension) then return calculateMean(tensor) end

	if (dimension ~= 1) and (dimension ~= 2) then error("Invalid dimension.") end

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)
	
	local size = dimensionSizeArray[dimension]
	
	local sumTensor = AqwamTensorLibrary:sum(tensor, dimension)
	
	local meanTensor = AqwamTensorLibrary:divide(sumTensor, size)

	return meanTensor

end

local function calculateStandardDeviation(tensor)

	local mean = calculateMean(tensor)

	local numberOfElements = #tensor * #tensor[1]

	local sum = 0
	
	for row, rowVector in ipairs(tensor) do

		for column, value in ipairs(rowVector) do

			sum = sum + (value - mean)

		end

	end

	local variance = (sum^2) / numberOfElements

	local standardDeviation = math.sqrt(variance)

	return standardDeviation, variance, mean

end

function AqwamTensorLibrary:standardDeviation(tensor, dimension)
	
	if (type(tensor) == "number") then return 0 end

	if (not dimension) then return calculateStandardDeviation(tensor) end
	
	if (dimension ~= 1) and (dimension ~= 2) then error("Invalid dimension.") end
	
	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)
	
	local size = dimensionSizeArray[dimension]

	local meanTensor = AqwamTensorLibrary:mean(tensor, dimension)

	local tensorSubtractedByMean = AqwamTensorLibrary:subtract(tensor, meanTensor)

	local squaredTensorSubtractedByMean = AqwamTensorLibrary:power(tensorSubtractedByMean, 2)

	local summedSquaredTensorSubtractedByMean = AqwamTensorLibrary:sum(squaredTensorSubtractedByMean, dimension)

	local varianceTensor = AqwamTensorLibrary:divide(summedSquaredTensorSubtractedByMean, size)

	local standardDeviationTensor = AqwamTensorLibrary:power(varianceTensor, 0.5)

	return standardDeviationTensor, varianceTensor, meanTensor

end

function AqwamTensorLibrary:generateTensorString(tensor)

	if tensor == nil then return "" end

	local numberOfRows = #tensor

	local numberOfColumns = #tensor[1]

	local columnWidths = {}

	-- Calculate maximum width for each column
	for column = 1, numberOfColumns, 1 do

		local maxWidth = 0

		for row = 1, numberOfRows do

			local cellWidth = string.len(tostring(tensor[row][column]))

			if (cellWidth > maxWidth) then

				maxWidth = cellWidth

			end

		end

		columnWidths[column] = maxWidth

	end

	local text = ""

	for row = 1, numberOfRows, 1 do

		text = text .. "{"

		for column = 1, numberOfColumns, 1 do

			local cellValue = tensor[row][column]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = columnWidths[column] - cellWidth + 1

			text = text .. string.rep(" ", padding) .. cellText
		end

		text = text .. " }\n"
	end

	return text

end

function AqwamTensorLibrary:printTensor(...)

	local text = "\n\n"

	local generatedText

	local tensors = {...}

	for tensorNumber = 1, #tensors, 1 do

		generatedText = AqwamTensorLibrary:generateTensorString(tensors[tensorNumber])

		text = text .. generatedText

		text = text .. "\n"

	end

	print(text)

end

function AqwamTensorLibrary:generateTensorWithCommaString(tensor)

	if tensor == nil then return "" end

	local numberOfRows = #tensor

	local numberOfColumns = #tensor[1]

	local columnWidths = {}

	-- Calculate maximum width for each column
	for column = 1, numberOfColumns, 1 do

		local maxWidth = 0

		for row = 1, numberOfRows, 1 do

			local cellWidth = string.len(tostring(tensor[row][column]))

			if (column < numberOfColumns) then

				cellWidth += 1

			end

			if (cellWidth > maxWidth) then

				maxWidth = cellWidth

			end

		end

		columnWidths[column] = maxWidth

	end

	local text = ""

	for row = 1, numberOfRows, 1 do

		text = text .. "{"

		for column = 1, numberOfColumns, 1 do

			local cellValue = tensor[row][column]

			local cellText = tostring(cellValue) 

			local cellWidth = string.len(cellText)

			local padding = columnWidths[column] - cellWidth + 1

			text = text .. string.rep(" ", padding) .. cellText

			if (column < numberOfColumns) then

				text = text .. ","

			end

		end

		text = text .. " }\n"
	end

	return text

end

function AqwamTensorLibrary:printTensorWithComma(...)

	local text = "\n\n"

	local generatedText

	local tensors = {...}

	for tensorNumber = 1, #tensors, 1 do

		generatedText = AqwamTensorLibrary:generateTensorWithCommaString(tensors[tensorNumber])

		text = text .. generatedText

		text = text .. "\n"

	end

	print(text)

end

function AqwamTensorLibrary:generatePortableTensorString(tensor)

	if tensor == nil then return "" end

	local numberOfRows = #tensor

	local numberOfColumns = #tensor[1]

	local columnWidths = {}

	-- Calculate maximum width for each column
	for column = 1, numberOfColumns, 1 do

		local maxWidth = 0

		for row = 1, numberOfRows, 1 do

			local cellWidth = string.len(tostring(tensor[row][column]))

			if (column < numberOfColumns) then

				cellWidth += 1

			end

			if (cellWidth > maxWidth) then

				maxWidth = cellWidth

			end

		end

		columnWidths[column] = maxWidth

	end

	local text = "{\n"

	for row = 1, numberOfRows, 1 do

		text = text .. "\t{"

		for column = 1, numberOfColumns, 1 do

			local cellValue = tensor[row][column]

			local cellText = tostring(cellValue) 

			local cellWidth = string.len(cellText)

			local padding = columnWidths[column] - cellWidth + 1

			text = text .. string.rep(" ", padding) .. cellText

			if (column < numberOfColumns) then

				text = text .. ","

			end

		end

		text = text .. " },\n"

	end

	text = text .. "}\n"

	return text

end

function AqwamTensorLibrary:printPortableTensor(...)

	local text = "\n\n"

	local generatedText

	local tensors = {...}

	for tensorNumber = 1, #tensors, 1 do

		generatedText = AqwamTensorLibrary:generatePortableTensorString(tensors[tensorNumber])

		text = text .. generatedText

		text = text .. "\n"

	end

	print(text)

end

local function rowConcatenate(tensor1, tensor2)

	local tensor1ColumnSize = #tensor1[1]
	local tensor2ColumnSize = #tensor2[1]

	if (tensor1ColumnSize ~= tensor2ColumnSize) then error("Incompatible Tensor Dimensions. Tensor 1 Has " .. tensor1ColumnSize .. " Column(s), Tensor 2 Has " .. tensor2ColumnSize .. " Column(s).") end

	local rowMiddleIndex = #tensor1

	local result = {}

	for row = 1, #tensor1, 1 do

		result[row] = {}

		for column = 1, #tensor1[1], 1 do

			result[row][column] = tensor1[row][column]

		end	

	end

	for row = 1, #tensor2, 1 do

		result[rowMiddleIndex + row] = {}

		for column = 1, #tensor2[1], 1 do

			result[rowMiddleIndex + row][column] = tensor2[row][column]

		end	

	end

	return result

end

local function columnConcatenate(tensor1, tensor2)

	local tensor1RowSize = #tensor1
	local tensor2RowSize = #tensor2

	if (tensor1RowSize ~= tensor2RowSize) then error("Incompatible Tensor Dimensions. Tensor 1 Has " .. tensor1RowSize .. " Row(s), Tensor 2 Has " .. tensor2RowSize .. " Row(s).") end

	local columnMiddleIndex = #tensor1[1]

	local result = {}

	for row = 1, #tensor1, 1 do

		result[row] = {}

		for column = 1, #tensor1[1], 1 do

			result[row][column] = tensor1[row][column]

		end	

	end

	for row = 1, #tensor2, 1 do

		for column = 1, #tensor2[1], 1 do

			result[row][columnMiddleIndex + column] = tensor2[row][column]

		end

	end

	return result

end

function AqwamTensorLibrary:rowConcatenate(...)

	local tensors = {...}

	local lastTensorIndex = #tensors
	local secondLastTensorIndex = lastTensorIndex - 1 

	local result = tensors[1]

	for i = 2, #tensors, 1 do

		local success = pcall(function()

			result = rowConcatenate(result, tensors[i])

		end)

		if (not success) then

			local text = generateArgumentErrorString(tensors, i - 1, i)

			error(text)

		end

	end

	return result

end

function AqwamTensorLibrary:columnConcatenate(...)

	local tensors = {...}

	local lastTensorIndex = #tensors
	local secondLastTensorIndex = lastTensorIndex - 1 

	local result = tensors[1]

	for i = 2, #tensors, 1 do

		local success = pcall(function()

			result = columnConcatenate(result, tensors[i])

		end)

		if (not success) then

			local text = generateArgumentErrorString(tensors, i - 1, i)

			error(text)

		end

	end

	return result

end

function AqwamTensorLibrary:concatenate(tensor1, tensor2, dimension)

	if (dimension == 1) then

		return rowConcatenate(tensor1, tensor2)

	elseif (dimension == 2) then

		return columnConcatenate(tensor1, tensor2)

	else

		error("Invalid dimension.")

	end

end

function AqwamTensorLibrary:applyFunction(functionToApply, ...)

	local tensorArray = {...}

	local numberOfTensors = #tensorArray
	
	local doAllTensorsHaveTheSameDimensionSizeArray

	--[[
		
		A single sweep is not enough to make sure that all tensors have the same dimension size arrays. So, we need to do it multiple times.
		
		Here's an example where the tensors' dimension size array will not match the others in a single sweep: {2, 3, 1}, {1, 3}, {5, 1, 1, 1}. 
		
		The first dimension size array needs to match with the third dimension size array, but can only look at the second dimension size array. 
		
		So, we need to propagate the third dimension size array to the nearby dimension size array so that it reaches the first dimension size array. 
		
		In this case, it would be the second dimension size array.
		
	--]]

	repeat 

		doAllTensorsHaveTheSameDimensionSizeArray = true

		for i = 1, (#tensorArray - 1), 1 do

			local tensor1 = tensorArray[i]

			local tensor2 = tensorArray[i + 1]

			local dimensionSizeArray1 = AqwamTensorLibrary:getDimensionSizeArray(tensor1)

			local dimensionSizeArray2 = AqwamTensorLibrary:getDimensionSizeArray(tensor2)

			if ((dimensionSizeArray1[1] ~= dimensionSizeArray2[1]) or (dimensionSizeArray1[2] ~= dimensionSizeArray2[2])) then doAllTensorsHaveTheSameDimensionSizeArray = false end

			tensorArray[i], tensorArray[i + 1] = broadcast(tensor1, tensor2, false)

		end

	until (doAllTensorsHaveTheSameDimensionSizeArray)
	
	local tensor = tensorArray[1]
	
	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)
	
	if (#dimensionSizeArray == 0) then return functionToApply(table.unpack(tensorArray)) end
	
	local numberOfRows = dimensionSizeArray[1]
	local numberOfColumns = dimensionSizeArray[2]

	local result = AqwamTensorLibrary:createTensor(dimensionSizeArray, true)

	local tensorValueArray = {}

	for row = 1, numberOfRows, 1 do

		for column = 1, numberOfColumns, 1 do

			for tensorIndex = 1, numberOfTensors, 1 do tensorValueArray[tensorIndex] = tensorArray[tensorIndex][row][column] end 

			result[row][column] = functionToApply(table.unpack(tensorValueArray))

		end	

	end

	return result

end

function AqwamTensorLibrary:findMaximumValue(tensor, dimension)
	
	if (not dimension) then

		local maximumValue = -math.huge

		for _, rowVector in ipairs(tensor) do
			
			for _, value in ipairs(rowVector) do
				
				maximumValue = math.max(maximumValue, value)
				
			end
			
		end

		return maximumValue

	elseif (dimension == 1) then

		local numberOfColumns = #tensor[1]
		
		local maximumVector = {}

		for j = 1, numberOfColumns do maximumVector[j] = -math.huge end

		for _, rowVector in ipairs(tensor) do
			
			for j, value in ipairs(rowVector) do maximumVector[j] = math.max(maximumVector[j], value) end
			
		end

		return {maximumVector}

	elseif (dimension == 2) then

		local maximumVector = {}

		for _, rowVector in ipairs(tensor) do
			
			local rowMaximumValue = math.max(table.unpack(rowVector))
			
			table.insert(maximumVector, {rowMaximumValue})
			
		end

		return maximumVector

	else
		
		error("Invalid dimension. Expected 1 or 2.")
		
	end
	
end

function AqwamTensorLibrary:findMaximumValueDimensionIndexArray(tensor)

	local dimensionIndexArray

	local maximumValue = -math.huge

	for row, rowVector in ipairs(tensor) do

		for column, value in ipairs(rowVector) do

			if (value > maximumValue) then

				maximumValue = value

				dimensionIndexArray = {row, column}

			end

		end

	end

	return dimensionIndexArray, maximumValue

end

function AqwamTensorLibrary:findMinimumValue(tensor, dimension)

	if (not dimension) then

		local minimumValue = math.huge

		for _, rowVector in ipairs(tensor) do
			
			for _, value in ipairs(rowVector) do
				
				minimumValue = math.min(minimumValue, value)
				
			end
			
		end

		return minimumValue

	elseif (dimension == 1) then

		local numberOfColumns = #tensor[1]

		local minimumVector = {}

		for j = 1, numberOfColumns do minimumVector[j] = math.huge end

		for _, rowVector in ipairs(tensor) do

			for j, value in ipairs(rowVector) do minimumVector[j] = math.min(minimumVector[j], value) end

		end

		return {minimumVector}

	elseif (dimension == 2) then

		local minimumVector = {}

		for _, rowVector in ipairs(tensor) do

			local rowMinimumValue = math.min(table.unpack(rowVector))

			table.insert(minimumVector, {rowMinimumValue})

		end

		return minimumVector

	else

		error("Invalid dimension. Expected 1 or 2.")

	end

end

function AqwamTensorLibrary:findMinimumValueDimensionIndexArray(tensor)

	local dimensionIndexArray

	local minimumValue = math.huge

	for row, rowVector in ipairs(tensor) do

		for column, value in ipairs(rowVector) do

			if (value < minimumValue) then

				minimumValue = value

				dimensionIndexArray = {row, column}

			end

		end

	end

	return dimensionIndexArray, minimumValue

end

function AqwamTensorLibrary:zScoreNormalization(tensor, dimension)

	local standardDeviationTensor, varianceTensor, meanTensor = AqwamTensorLibrary:standardDeviation(tensor, dimension)

	local zScoreTensor = AqwamTensorLibrary:subtract(tensor, meanTensor)

	zScoreTensor = AqwamTensorLibrary:divide(zScoreTensor, standardDeviationTensor)

	return zScoreTensor, standardDeviationTensor, varianceTensor, meanTensor

end

function AqwamTensorLibrary:extractRows(tensor, startingRowIndex, endingRowIndex)

	if (endingRowIndex == nil) then endingRowIndex = #tensor end

	if (startingRowIndex <= 0) then error("The starting row index must be greater than 0.") end 

	if (endingRowIndex <= 0) then error("The ending row index must be greater than 0.") end

	local numberOfRows = #tensor

	local result = {}

	for row = startingRowIndex, endingRowIndex do

		table.insert(result, tensor[row])

	end

	return result

end

function AqwamTensorLibrary:extractColumns(tensor, startingColumnIndex, endingColumnIndex)

	if (endingColumnIndex == nil) then endingColumnIndex = #tensor end

	if (startingColumnIndex <= 0) then error("The starting column index must be greater than 0.") end 

	if (endingColumnIndex <= 0) then error("The ending column index must be greater than 0.") end

	local numberOfRows = #tensor

	local result = {}

	for row = 1, numberOfRows, 1 do

		result[row] = {}

		for column = startingColumnIndex, endingColumnIndex do 

			table.insert(result[row], tensor[row][column])

		end

	end

	return result

end

function AqwamTensorLibrary:extract(tensor, originDimensionIndexArray, targetDimensionIndexArray)
	
	local rowOriginIndex = originDimensionIndexArray[1]
	
	local rowTargetIndex = targetDimensionIndexArray[1]
	
	local columnOriginIndex = originDimensionIndexArray[2]
	
	local columnTargetIndex = targetDimensionIndexArray[2]
	
	local result = {}

	for row = rowOriginIndex, rowTargetIndex, 1 do

		result[row] = {}

		for column = columnOriginIndex, columnTargetIndex, 1 do 

			table.insert(result[row], tensor[row][column])

		end

	end

	return result
	
end

function AqwamTensorLibrary:copy(tensor)

	return deepCopyTable(tensor)

end

function AqwamTensorLibrary:minor(tensor, row, column)

	local dimensionSize = #tensor

	local minor = {}

	for i = 1, dimensionSize - 1 do

		minor[i] = {}

		for j = 1, dimensionSize - 1 do

			local mRow = (i < row and i) or (i + 1)

			local mColumn = (j < column and j) or (j + 1)

			minor[i][j] = tensor[mRow][mColumn]

		end

	end

	return minor

end

function  AqwamTensorLibrary:cofactor(tensor, row, column)

	local minor =  AqwamTensorLibrary:minor(tensor, row, column)

	local sign = (((row + column) % 2 == 0) and 1) or -1

	return sign * AqwamTensorLibrary:determinant(minor)

end

function AqwamTensorLibrary:determinant(tensor)

	local dimensionSize = #tensor

	if (dimensionSize == 1) then

		return tensor[1][1]

	elseif (dimensionSize == 2) then

		return tensor[1][1] * tensor[2][2] - tensor[1][2] * tensor[2][1]

	else

		local determinant = 0

		for i = 1, dimensionSize do

			local cofactor =  AqwamTensorLibrary:cofactor(tensor, 1, i)

			determinant = determinant + tensor[1][i] * cofactor

		end

		return determinant

	end

end

function AqwamTensorLibrary:inverse(tensor)

	if (#tensor ~= #tensor[1]) then return nil end

	local dimensionSize = #tensor

	local determinant = AqwamTensorLibrary:determinant(tensor)

	if (determinant == 0) then

		return nil -- tensor is not invertible

	elseif (dimensionSize == 1) then

		return {{1 / determinant}}

	else

		local adjugate = {}

		for i = 1, dimensionSize do

			adjugate[i] = {}

			for j = 1, dimensionSize do

				local sign = ((i + j) % 2 == 0) and 1 or -1

				local cof = AqwamTensorLibrary:cofactor(tensor, i, j)

				adjugate[i][j] = sign * cof

			end

		end

		local inverseTensor = AqwamTensorLibrary:transpose(adjugate)

		for i = 1, dimensionSize do

			for j = 1, dimensionSize do

				inverseTensor[i][j] = inverseTensor[i][j] / determinant

			end

		end

		return inverseTensor

	end

end

function AqwamTensorLibrary:isTensor(tensor)

	local tensorCheck

	local notIndexNumberCheck

	local itIsATensor

	tensorCheck = pcall(function()

		local test = tensor[1][1]

	end)

	notIndexNumberCheck = pcall(function()

		local test = tensor[1][1][1]

	end)

	itIsATensor = (tensorCheck) and (not notIndexNumberCheck)

	return itIsATensor 

end

function AqwamTensorLibrary:findNanValue(tensor)

	for row = 1, #tensor, 1 do

		for column = 1, #tensor[1] do

			local value = tensor[row][column]

			if (value ~= value) then return {row, column} end

		end

	end

	return nil

end

function AqwamTensorLibrary:findValue(tensor, valueToFind)

	for row = 1, #tensor, 1 do

		for column = 1, #tensor[1] do 

			if (tensor[row][column] == valueToFind) then return {row, column} end

		end

	end

	return nil

end

function AqwamTensorLibrary:setValue(tensor, value, dimensionIndexArray)
	
	local rowIndex = dimensionIndexArray[1]
	
	local columnIndex = dimensionIndexArray[2]

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	if (rowIndex < 1) or (rowIndex > dimensionSizeArray[1]) or (columnIndex < 1) or (columnIndex > dimensionSizeArray[2]) then error("Attempting to set a value that is out of bounds.") end

	tensor[rowIndex][columnIndex] = value

end

function AqwamTensorLibrary:getValue(tensor, dimensionIndexArray)
	
	local rowIndex = dimensionIndexArray[1]

	local columnIndex = dimensionIndexArray[2]

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	if (rowIndex < 1) or (rowIndex > dimensionSizeArray[1]) or (columnIndex < 1) or (columnIndex > dimensionSizeArray[2]) then error("Attempting to get a value that is out of bounds.") end

	return tensor[rowIndex][columnIndex]

end

function AqwamTensorLibrary:flip(tensor, dimension)
	
	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)
	
	local numberOfRows = dimensionSizeArray[1]
	
	local numberOfColumns = dimensionSizeArray[2]
	
	local resultTensor = {}
	
	if (dimension == 1) then
		
		for i = 1, numberOfRows, 1 do
			
			resultTensor[i] = {}
			
			for j = 1, numberOfColumns, 1 do
				
				resultTensor[i][j] = tensor[(numberOfRows - i) + 1][j]
				
			end
			
		end
		
	elseif (dimension == 2) then
		
		for i = 1, numberOfRows, 1 do

			resultTensor[i] = {}
			
			for j = 1, numberOfColumns, 1 do
				
				resultTensor[i][j] = tensor[i][(numberOfColumns - j) + 1]
				
			end

		end
		
	else
		
		error("Invalid dimension.")
		
	end

	return resultTensor
	
end

return AqwamTensorLibrary
