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

	local hasSamenumberOfRows = (tensor1Rows == tensor2Rows)

	local hasSamenumberOfColumns = (tensor1Columns == tensor2Columns)

	local hasSameDimension = hasSamenumberOfRows and hasSamenumberOfColumns

	local isTensor1IsLargerInOneDimension = ((tensor1Rows > 1) and hasSamenumberOfColumns and (tensor2Rows == 1)) or ((tensor1Columns > 1) and hasSamenumberOfRows and (tensor2Columns == 1))

	local isTensor2IsLargerInOneDimension = ((tensor2Rows > 1) and hasSamenumberOfColumns and (tensor1Rows == 1)) or ((tensor2Columns > 1) and hasSamenumberOfRows and (tensor1Columns == 1))

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

function AqwamTensorLibrary:expand(tensor, targetnumberOfRows, targetnumberOfColumns)

	local resultTensor = {}

	local isTensornumberOfRowsEqualToOne = (#tensor == 1)

	local isTensornumberOfColumnsEqualToOne = (#tensor[1] == 1)

	if (isTensornumberOfRowsEqualToOne) and (not isTensornumberOfColumnsEqualToOne) then

		for rowIndex = 1, targetnumberOfRows, 1 do
			
			local resultVector = {}

			for columnIndex = 1, targetnumberOfColumns, 1 do resultVector[columnIndex] = tensor[1][columnIndex] end
			
			resultTensor[rowIndex] = resultVector

		end

	elseif (not isTensornumberOfRowsEqualToOne) and (isTensornumberOfColumnsEqualToOne) then

		for rowIndex = 1, targetnumberOfRows, 1 do

			local resultVector = {}

			for columnIndex = 1, targetnumberOfColumns, 1 do resultVector[columnIndex] = tensor[rowIndex][1] end
			
			resultTensor[rowIndex] = resultVector

		end

	elseif (isTensornumberOfRowsEqualToOne) and (isTensornumberOfColumnsEqualToOne) then

		for rowIndex = 1, targetnumberOfRows, 1 do

			local resultVector = {}

			for columnIndex = 1, targetnumberOfColumns, 1 do resultVector[columnIndex] = tensor[1][1] end
			
			resultTensor[rowIndex] = resultVector

		end

	end

	return resultTensor

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
	
	local isAScalar = (type(value) ~= "table")

	if (isAScalar) then

		return {{value}}

	else

		return value

	end

end

local function onDotProductError(tensor1Column, tensor2Row)

	local errorMessage = "Incompatible tensor dimensions: " .. tensor1Column .. " column(s), " .. tensor2Row .. " row(s)."

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

	local resultTensor = {}

	local tensor1Row = #tensor1
	local tensor1Column = #tensor1[1]
	local tensor2Column = #tensor2[1]
	
	local tensor1Array

	checkIfCanDotProduct(tensor1, tensor2)

	for rowIndex = 1, tensor1Row, 1 do
		
		local resultVector = {}
		
		tensor1Array = tensor1[rowIndex]

		for columnIndex = 1, tensor2Column, 1 do

			local sum = 0

			for i = 1, tensor1Column do sum = sum + (tensor1Array[i] * tensor2[i][columnIndex]) end

			resultVector[columnIndex] = sum

		end
		
		resultTensor[rowIndex] = resultVector

	end

	return resultTensor

end

local function generateArgumentErrorString(tensorArray, firstTensorIndex, secondTensorIndex)

	local text1 = "Argument " .. firstTensorIndex .. " and " .. secondTensorIndex .. " are incompatible! "

	local text2 = "(" ..  #tensorArray[firstTensorIndex] .. ", " .. #tensorArray[firstTensorIndex][1] .. ") and " .. "(" ..  #tensorArray[secondTensorIndex] .. ", " .. #tensorArray[secondTensorIndex][1] .. ")"

	local text = text1 .. text2

	return text

end

local function applyFunctionUsingOneTensor(functionToApply, tensor)

	local resultTensor = {}

	local resultVector

	for rowIndex, rowVector in ipairs(tensor) do

		resultVector = {}

		for columnIndex, value in ipairs(rowVector) do

			resultVector[columnIndex] = functionToApply(value)

		end

		resultTensor[rowIndex] = resultVector

	end

	return resultTensor

end

local function applyFunctionUsingTwoTensors(functionToApply, tensor1, tensor2)

	if (#tensor1 ~= #tensor2) or (#tensor1[1] ~= #tensor2[1]) then error("Incompatible Dimensions! (" .. #tensor1 .." x " .. #tensor1[1] .. ") and (" .. #tensor2 .. " x " .. #tensor2[1] .. ")") end

	local resultTensor = {}

	local resultVector
	
	local rowVector2

	for rowIndex, rowVector1 in ipairs(tensor1) do
		
		rowVector2 = tensor2[rowIndex]

		resultVector = {}
		
		for columnIndex, value in ipairs(rowVector1) do

			resultVector[columnIndex] = functionToApply(value, rowVector2[columnIndex])

		end

		resultTensor[rowIndex] = resultVector

	end

	return resultTensor

end

local function applyFunctionWhenTheFirstValueIsAScalar(functionToApply, scalar, tensor)

	local resultTensor = {}
	
	local resultVector
	
	for rowIndex, rowVector in ipairs(tensor) do
		
		resultVector = {}
		
		for columnIndex, value in ipairs(rowVector) do
			
			resultVector[columnIndex] = functionToApply(scalar, value)
			
		end
		
		resultTensor[rowIndex] = resultVector
		
	end

	return resultTensor

end

local function applyFunctionWhenTheSecondValueIsAScalar(functionToApply, tensor, scalar)

	local resultTensor = {}

	local resultVector

	for rowIndex, rowVector in ipairs(tensor) do

		resultVector = {}

		for columnIndex, value in ipairs(rowVector) do

			resultVector[columnIndex] = functionToApply(value, scalar)

		end

		resultTensor[rowIndex] = resultVector

	end

	return resultTensor

end

local function applyFunctionUsingMultipleTensors(functionToApply, ...)

	local tensorArray = {...}

	local numberOfTensors = #tensorArray

	local tensor = tensorArray[1]

	if (numberOfTensors == 1) then 

		if (type(tensor) == "table") then

			return applyFunctionUsingOneTensor(functionToApply, tensor) 

		else

			return functionToApply(tensor)

		end

	end

	for i = 2, numberOfTensors, 1 do

		local otherTensor = tensorArray[i]

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

	local tensorArray = {...}

	local resultTensor = tensorArray[1]
	
	local secondTensor

	resultTensor = convertToTensorIfScalar(resultTensor)

	for i = 2, #tensorArray, 1 do

		resultTensor = convertToTensorIfScalar(resultTensor)
		
		secondTensor = convertToTensorIfScalar(tensorArray[i])

		resultTensor = dotProduct(resultTensor, secondTensor)

	end

	return resultTensor

end

function AqwamTensorLibrary:createIdentityTensor(dimensionSizeArray, value)

	if (#dimensionSizeArray ~= 2) then error("Invalid dimension size array.") end

	local numberOfRows = dimensionSizeArray[1]

	local numberOfColumns = dimensionSizeArray[2]

	local resultTensor = {}
	
	value = value or 1

	for rowIndex = 1, numberOfRows, 1 do
		
		local rowVector = table.create(numberOfColumns, 0) 
		
		rowVector[rowIndex] = value
		
		resultTensor[rowIndex] = rowVector
		
	end

	return resultTensor

end

function AqwamTensorLibrary:createTensor(dimensionSizeArray, allValues)

	local numberOfRows = dimensionSizeArray[1]

	local numberOfColumns = dimensionSizeArray[2]
	
	allValues = allValues or 0

	local resultTensor = {}
	
	for rowIndex = 1, numberOfRows, 1 do resultTensor[rowIndex] = table.create(numberOfColumns, allValues) end

	return resultTensor

end

function AqwamTensorLibrary:createRandomNormalTensor(dimensionSizeArray, mean, standardDeviation)

	local numberOfRows = dimensionSizeArray[1]

	local numberOfColumns = dimensionSizeArray[2]

	local resultTensor = {}

	mean = mean or 0

	standardDeviation = standardDeviation or 1

	for rowIndex = 1, numberOfRows, 1 do
		
		local resultVector = {}

		for columnIndex = 1, numberOfColumns do

			local randomNumber1 = math.random()

			local randomNumber2 = math.random()

			local zScore = math.sqrt(-2 * math.log(randomNumber1)) * math.cos(2 * math.pi * randomNumber2)

			resultVector[columnIndex] = (zScore * standardDeviation) + mean

		end
		
		resultTensor[rowIndex] = resultVector
		
	end

	return resultTensor

end

function AqwamTensorLibrary:createRandomUniformTensor(dimensionSizeArray, minimumValue, maximumValue)

	local numberOfRows = dimensionSizeArray[1]

	local numberOfColumns = dimensionSizeArray[2]

	local resultTensor = {}
	
	if (minimumValue) and (maximumValue) then
		
		if (minimumValue >= maximumValue) then error("The minimum value cannot exceed the maximum value.") end
		
		local rangeValue = maximumValue - minimumValue
		
		for rowIndex = 1, numberOfRows, 1 do

			local resultVector = {}

			for columnIndex = 1, numberOfColumns, 1 do

				resultVector[columnIndex] = minimumValue + (math.random() * rangeValue)

			end
			
			resultTensor[rowIndex] = resultVector

		end
		
	elseif (not minimumValue) and (maximumValue) then
		
		if (maximumValue <= 0) then error("The maximum value cannot be less than or equal to zero.") end

		for rowIndex = 1, numberOfRows, 1 do

			local resultVector = {}

			for columnIndex = 1, numberOfColumns, 1 do

				resultVector[columnIndex] = math.random() * maximumValue

			end
			
			resultTensor[rowIndex] = resultVector

		end

	elseif (minimumValue) and (not maximumValue) then
		
		if (minimumValue >= 0) then error("The minimum value cannot be greater than or equal to zero.") end

		for rowIndex = 1, numberOfRows, 1 do

			local resultVector = {}

			for columnIndex = 1, numberOfColumns, 1 do

				resultVector[columnIndex] = math.random() * minimumValue

			end	
			
			resultTensor[rowIndex] = resultVector

		end
		
	elseif (not minimumValue) and (not maximumValue) then

		for rowIndex = 1, numberOfRows, 1 do

			local resultVector = {}

			for columnIndex = 1, numberOfColumns, 1 do

				resultVector[columnIndex] = (math.random() * 2) - 1

			end
			
			resultTensor[rowIndex] = resultVector

		end

	end

	return resultTensor

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
	
	local resultTensor = {}
	
	for columnIndex = 1, numberOfColumns, 1 do
		
		local resultVector = {}
		
		for rowIndex = 1, numberOfRows, 1 do
			
			resultVector[rowIndex] = tensor[rowIndex][columnIndex]
			
		end
		
		resultTensor[columnIndex] = resultVector
		
	end

	return resultTensor

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

	local resultVector = {}
	
	for _, rowVector in ipairs(tensor) do
		
		for columnIndex, value in ipairs(rowVector) do
			
			resultVector[columnIndex] = resultVector[columnIndex] + value
			
		end
		
	end
	
	local resultTensor = {resultVector}

	return resultTensor

end

local function columnSum(tensor)

	local numberOfRows = #tensor
	
	local columnSumArray = {}

	for rowIndex, rowVector in ipairs(tensor) do

		for columnIndex, value in ipairs(rowVector) do

			columnSumArray[rowIndex] = columnSumArray[rowIndex] + value

		end

	end
	
	local resultTensor = {}
	
	for rowIndex = 1, numberOfRows, 1 do resultTensor[rowIndex] = {columnSumArray[rowIndex]} end

	return resultTensor

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

	local sumSquaredDifference = 0
	
	for rowIndex, rowVector in ipairs(tensor) do

		for columnIndex, value in ipairs(rowVector) do
			
			local difference = value - mean
			
			local squaredDifference = math.pow(difference, 2)

			sumSquaredDifference = sumSquaredDifference + squaredDifference

		end

	end

	local variance = sumSquaredDifference / numberOfElements

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

	if (not tensor) then return "" end

	local numberOfRows = #tensor

	local numberOfColumns = #tensor[1]

	local columnWidths = {}

	-- Calculate maximum width for each columnIndex
	for columnIndex = 1, numberOfColumns, 1 do

		local maxWidth = 0

		for rowIndex = 1, numberOfRows do

			local cellWidth = string.len(tostring(tensor[rowIndex][columnIndex]))

			if (cellWidth > maxWidth) then

				maxWidth = cellWidth

			end

		end

		columnWidths[columnIndex] = maxWidth

	end

	local text = ""

	for rowIndex = 1, numberOfRows, 1 do

		text = text .. "{"

		for columnIndex = 1, numberOfColumns, 1 do

			local cellValue = tensor[rowIndex][columnIndex]

			local cellText = tostring(cellValue)

			local cellWidth = string.len(cellText)

			local padding = columnWidths[columnIndex] - cellWidth + 1

			text = text .. string.rep(" ", padding) .. cellText
		end

		text = text .. " }\n"
	end

	return text

end

function AqwamTensorLibrary:printTensor(...)

	local text = "\n\n"

	local generatedText

	local tensorArray = {...}

	for tensorNumber = 1, #tensorArray, 1 do

		generatedText = AqwamTensorLibrary:generateTensorString(tensorArray[tensorNumber])

		text = text .. generatedText

		text = text .. "\n"

	end

	print(text)

end

function AqwamTensorLibrary:generateTensorWithCommaString(tensor)

	if (not tensor) then return "" end

	local numberOfRows = #tensor

	local numberOfColumns = #tensor[1]

	local columnWidths = {}

	-- Calculate maximum width for each columnIndex
	for columnIndex = 1, numberOfColumns, 1 do

		local maxWidth = 0

		for rowIndex = 1, numberOfRows, 1 do

			local cellWidth = string.len(tostring(tensor[rowIndex][columnIndex]))

			if (columnIndex < numberOfColumns) then

				cellWidth += 1

			end

			if (cellWidth > maxWidth) then

				maxWidth = cellWidth

			end

		end

		columnWidths[columnIndex] = maxWidth

	end

	local text = ""

	for rowIndex = 1, numberOfRows, 1 do

		text = text .. "{"

		for columnIndex = 1, numberOfColumns, 1 do

			local cellValue = tensor[rowIndex][columnIndex]

			local cellText = tostring(cellValue) 

			local cellWidth = string.len(cellText)

			local padding = columnWidths[columnIndex] - cellWidth + 1

			text = text .. string.rep(" ", padding) .. cellText

			if (columnIndex < numberOfColumns) then

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

	local tensorArray = {...}

	for tensorNumber = 1, #tensorArray, 1 do

		generatedText = AqwamTensorLibrary:generateTensorWithCommaString(tensorArray[tensorNumber])

		text = text .. generatedText

		text = text .. "\n"

	end

	print(text)

end

function AqwamTensorLibrary:generatePortableTensorString(tensor)

	if (not tensor) then return "" end

	local numberOfRows = #tensor

	local numberOfColumns = #tensor[1]

	local columnWidths = {}

	-- Calculate maximum width for each columnIndex
	for columnIndex = 1, numberOfColumns, 1 do

		local maxWidth = 0

		for rowIndex = 1, numberOfRows, 1 do

			local cellWidth = string.len(tostring(tensor[rowIndex][columnIndex]))

			if (columnIndex < numberOfColumns) then

				cellWidth += 1

			end

			if (cellWidth > maxWidth) then

				maxWidth = cellWidth

			end

		end

		columnWidths[columnIndex] = maxWidth

	end

	local text = "{\n"

	for rowIndex = 1, numberOfRows, 1 do

		text = text .. "\t{"

		for columnIndex = 1, numberOfColumns, 1 do

			local cellValue = tensor[rowIndex][columnIndex]

			local cellText = tostring(cellValue) 

			local cellWidth = string.len(cellText)

			local padding = columnWidths[columnIndex] - cellWidth + 1

			text = text .. string.rep(" ", padding) .. cellText

			if (columnIndex < numberOfColumns) then

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

	local tensorArray = {...}

	for tensorNumber = 1, #tensorArray, 1 do

		generatedText = AqwamTensorLibrary:generatePortableTensorString(tensorArray[tensorNumber])

		text = text .. generatedText

		text = text .. "\n"

	end

	print(text)

end

local function rowConcatenate(tensor1, tensor2)

	local tensor1numberOfColumns = #tensor1[1]
	
	local tensor2numberOfColumns = #tensor2[1]

	if (tensor1numberOfColumns ~= tensor2numberOfColumns) then error("Incompatible tensor dimensions. Tensor 1 Has " .. tensor1numberOfColumns .. " column(s), Tensor 2 has " .. tensor2numberOfColumns .. " column(s).") end
	
	local tensor1numberOfRows = #tensor1
	
	local tensor2numberOfRows = #tensor2

	local rowMiddleIndex = tensor1numberOfRows

	local resultTensor = {}

	for rowIndex = 1, tensor1numberOfRows, 1 do
		
		local resultVector = {}

		for columnIndex = 1, tensor1numberOfColumns, 1 do

			resultVector[columnIndex] = tensor1[rowIndex][columnIndex]

		end
		
		resultTensor[rowIndex] = resultVector

	end

	for rowIndex = 1, tensor2numberOfRows, 1 do
		
		local resultVector = {}

		for columnIndex = 1, tensor2numberOfColumns, 1 do

			resultVector[columnIndex] = tensor2[rowIndex][columnIndex]

		end
		
		resultTensor[rowMiddleIndex + rowIndex] = resultVector

	end

	return resultTensor

end

local function columnConcatenate(tensor1, tensor2)

	local tensor1numberOfRows = #tensor1
	
	local tensor2numberOfRows = #tensor2

	if (tensor1numberOfRows ~= tensor2numberOfRows) then error("Incompatible tensor dimensions. Tensor 1 has " .. tensor1numberOfRows .. " row(s), Tensor 2 has " .. tensor2numberOfRows .. " row(s).") end
	
	local tensor1numberOfColumns = #tensor1[1]
	
	local tensor2numberOfColumns = #tensor2[1]

	local columnMiddleIndex = #tensor1[1]

	local resultTensor = {}

	for rowIndex = 1, tensor1numberOfRows, 1 do
		
		local resultVector = {}
		
		local columnIndex = 1

		for columnIndex = 1, tensor1numberOfColumns, 1 do

			resultVector[columnIndex] = tensor1[rowIndex][columnIndex]
			
			columnIndex = columnIndex + 1

		end
		
		for columnIndex = 1, tensor2numberOfColumns, 1 do

			resultVector[columnIndex] = tensor2[rowIndex][columnIndex]
			
			columnIndex = columnIndex + 1

		end
		
		resultTensor[rowIndex] = resultVector

	end

	return resultTensor

end

function AqwamTensorLibrary:rowConcatenate(...)

	local tensorArray = {...}

	local lastTensorIndex = #tensorArray
	local secondLastTensorIndex = lastTensorIndex - 1 

	local resultTensor = tensorArray[1]

	for i = 2, #tensorArray, 1 do

		local success = pcall(function()

			resultTensor = rowConcatenate(resultTensor, tensorArray[i])

		end)

		if (not success) then

			local text = generateArgumentErrorString(tensorArray, i - 1, i)

			error(text)

		end

	end

	return resultTensor

end

function AqwamTensorLibrary:columnConcatenate(...)

	local tensorArray = {...}

	local lastTensorIndex = #tensorArray
	local secondLastTensorIndex = lastTensorIndex - 1 

	local resultTensor = tensorArray[1]

	for i = 2, #tensorArray, 1 do

		local success = pcall(function()

			resultTensor = columnConcatenate(resultTensor, tensorArray[i])

		end)

		if (not success) then

			local text = generateArgumentErrorString(tensorArray, i - 1, i)

			error(text)

		end

	end

	return resultTensor

end

function AqwamTensorLibrary:concatenate(tensor1, tensor2, dimension)
	
	if (type(tensor1) == "number") then tensor1 = {{tensor1}} end
	
	if (type(tensor2) == "number") then tensor2 = {{tensor2}} end

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

	local resultTensor = {}

	local tensorValueArray = {}

	for rowIndex = 1, numberOfRows, 1 do
		
		local resultVector = {}

		for columnIndex = 1, numberOfColumns, 1 do

			for tensorIndex = 1, numberOfTensors, 1 do tensorValueArray[tensorIndex] = tensorArray[tensorIndex][rowIndex][columnIndex] end 

			resultVector[columnIndex] = functionToApply(table.unpack(tensorValueArray))

		end
		
		resultTensor[rowIndex] = resultVector

	end

	return resultTensor

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

	for rowIndex, rowVector in ipairs(tensor) do

		for columnIndex, value in ipairs(rowVector) do

			if (value > maximumValue) then

				maximumValue = value

				dimensionIndexArray = {rowIndex, columnIndex}

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

	for rowIndex, rowVector in ipairs(tensor) do

		for columnIndex, value in ipairs(rowVector) do

			if (value < minimumValue) then

				minimumValue = value

				dimensionIndexArray = {rowIndex, columnIndex}

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

	if (not endingRowIndex) then endingRowIndex = #tensor end

	if (startingRowIndex <= 0) then error("The starting rowIndex index must be greater than 0.") end 

	if (endingRowIndex <= 0) then error("The ending rowIndex index must be greater than 0.") end

	local numberOfRows = #tensor

	local resultTensor = {}

	for rowIndex = startingRowIndex, endingRowIndex do

		table.insert(resultTensor, tensor[rowIndex])

	end

	return resultTensor

end

function AqwamTensorLibrary:extractColumns(tensor, startingColumnIndex, endingColumnIndex)

	if (not endingColumnIndex) then endingColumnIndex = #tensor[1] end

	if (startingColumnIndex <= 0) then error("The starting columnIndex index must be greater than 0.") end 

	if (endingColumnIndex <= 0) then error("The ending columnIndex index must be greater than 0.") end

	local numberOfRows = #tensor

	local resultTensor = {}

	for rowIndex = 1, numberOfRows, 1 do

		resultTensor[rowIndex] = {}

		for columnIndex = startingColumnIndex, endingColumnIndex do 

			table.insert(resultTensor[rowIndex], tensor[rowIndex][columnIndex])

		end

	end

	return resultTensor

end

function AqwamTensorLibrary:extract(tensor, originDimensionIndexArray, targetDimensionIndexArray)
	
	local rowOriginIndex = originDimensionIndexArray[1]
	
	local rowTargetIndex = targetDimensionIndexArray[1]
	
	local columnOriginIndex = originDimensionIndexArray[2]
	
	local columnTargetIndex = targetDimensionIndexArray[2]
	
	local resultTensor = {}

	for rowIndex = rowOriginIndex, rowTargetIndex, 1 do
		
		local resultVector = {}

		for columnIndex = columnOriginIndex, columnTargetIndex, 1 do 

			table.insert(resultVector, tensor[rowIndex][columnIndex])

		end
		
		resultTensor[rowIndex] = resultVector

	end

	return resultTensor
	
end

function AqwamTensorLibrary:copy(tensor)

	return deepCopyTable(tensor)

end

function AqwamTensorLibrary:minor(tensor, rowIndex, columnIndex)

	local dimensionSize = #tensor
	
	local dimensionSizeMinusOne = dimensionSize - 1
	
	local minorTensor = {}

	for i = 1, dimensionSizeMinusOne, 1 do
		
		local minorVector = {}

		for j = 1, dimensionSizeMinusOne, 1 do

			local mRow = (i < rowIndex and i) or (i + 1)

			local mColumn = (j < columnIndex and j) or (j + 1)

			minorVector[j] = tensor[mRow][mColumn]

		end
		
		minorTensor[i] = minorVector

	end

	return minorTensor

end

function  AqwamTensorLibrary:cofactor(tensor, rowIndex, columnIndex)

	local minor =  AqwamTensorLibrary:minor(tensor, rowIndex, columnIndex)

	local sign = (((rowIndex + columnIndex) % 2 == 0) and 1) or -1
	
	local determinant =AqwamTensorLibrary:determinant(minor)
	
	local cofactor = sign * determinant

	return cofactor 

end

function AqwamTensorLibrary:determinant(tensor)

	local dimensionSize = #tensor

	if (dimensionSize == 1) then

		return tensor[1][1]

	elseif (dimensionSize == 2) then

		return tensor[1][1] * tensor[2][2] - tensor[1][2] * tensor[2][1]

	else

		local determinant = 0

		for i = 1, dimensionSize, 1 do

			local cofactor = AqwamTensorLibrary:cofactor(tensor, 1, i)

			determinant = determinant + (tensor[1][i] * cofactor)

		end

		return determinant

	end

end

local function determinantInverse(tensor)
	
	local dimensionSize = #tensor

	local determinant = AqwamTensorLibrary:determinant(tensor)

	if (determinant == 0) then return end -- tensor is not invertible

	if (dimensionSize == 1) then return {{1 / determinant}} end

	local adjugateTensor = {}

	for i = 1, dimensionSize, 1 do
		
		local adjugateVector = {}

		for j = 1, dimensionSize, 1 do

			local cofactor = AqwamTensorLibrary:cofactor(tensor, i, j)

			adjugateVector[j] = cofactor

		end
		
		adjugateTensor[i] = adjugateVector

	end

	local inverseTensor = AqwamTensorLibrary:transpose(adjugateTensor)

	for i = 1, dimensionSize, 1 do
		
		local inverseVector = inverseTensor[i]

		for j = 1, dimensionSize, 1 do

			inverseVector[j] = inverseVector[j] / determinant

		end

	end

	return inverseTensor
	
end

local function luDecompositionInverse(tensor)

	local dimensionSize = #tensor

	-- Create augmented tensor [A | I].
	
	local augmentedTensor = {}

	for i = 1, dimensionSize, 1 do
		
		local augmentedVector = {}

		-- Copy A tensor.
		
		for j = 1, dimensionSize, 1 do
			
			augmentedVector[j] = tensor[i][j]

		end

		-- Append identity I tensor.
		
		for j = 1, dimensionSize, 1 do

			augmentedVector[dimensionSize + j] = (i == j) and 1 or 0

		end
		
		augmentedTensor[i] = augmentedVector

	end

	-- Forward elimination with partial pivoting.
	
	for k = 1, dimensionSize, 1 do

		-- 1. Find pivot.
		
		local maximumValue = math.abs(augmentedTensor[k][k])
		
		local maximumRowIndex = k

		for i = k + 1, dimensionSize, 1 do

			local value = math.abs(augmentedTensor[i][k])

			if (value > maximumValue) then

				maximumValue = value
				
				maximumRowIndex = i

			end

		end

		if (maximumValue <= 0) then return end -- Tensor is singular.

		-- 2. Swap Rows if needed.
		
		if (maximumRowIndex ~= k) then

			local augmentedVector = augmentedTensor[k]
			
			augmentedTensor[k] = augmentedTensor[maximumRowIndex]
			
			augmentedTensor[maximumRowIndex] = augmentedVector

		end

		-- 3. Eliminate columnIndex.
		
		for i = k + 1, dimensionSize, 1 do

			local factor = augmentedTensor[i][k] / augmentedTensor[k][k]
			
			local augmentedVector = augmentedTensor[i]

			for j = k, (dimensionSize * 2), 1 do

				augmentedVector[j] = augmentedVector[j] - (factor * augmentedTensor[k][j])

			end

		end

	end

	-- Backward substitution.
	
	for k = dimensionSize, 1, -1 do

		-- Normalize pivot rowIndex.
		
		local pivot = augmentedTensor[k][k]
		
		local augmentedVector = augmentedTensor[k]

		for j = k, (dimensionSize * 2), 1 do

			augmentedVector[j] = augmentedVector[j] / pivot

		end

		-- Eliminate upwards.
		
		for i = k - 1, 1, -1 do

			local factor = augmentedTensor[i][k]
			
			local augmentedVector = augmentedTensor[i]

			for j = k, (dimensionSize * 2), 1 do

				augmentedVector[j] = augmentedVector[j] - factor * augmentedTensor[k][j]

			end

		end

	end

	-- Extract resultTensor (right half of augmented tensor).
	
	local inverseTensor = {}

	for i = 1, dimensionSize, 1 do
		
		local inverseVector = {}

		for j = 1, dimensionSize, 1 do

			inverseVector[j] = augmentedTensor[i][dimensionSize + j]

		end
		
		inverseTensor[i] = inverseVector

	end

	return inverseTensor

end

function AqwamTensorLibrary:inverse(tensor, method)
	
	local numberOfRows = #tensor
	
	local numberOfColumns = #tensor[1]

	if (numberOfRows ~= numberOfColumns) or (numberOfRows == 0) or (numberOfColumns == 0) then return end
	
	if (not method) then
		
		-- LU decomposition inverse has a time complexity of O(n^3). Determinant inverse has a time complexity of O(n!).
		
		-- When dimensionSize is 6, the time complexity for the LU decomposition inverse is O(216). Meanwhile the determinant inverse is O(720).
		
		local isDimensionSizeLargerOrEqualToSix = (numberOfRows >= 6)
		
		method = (isDimensionSizeLargerOrEqualToSix and "LUDecomposition") or "Determinant"
		
	end
	
	if (method == "Determinant") then return determinantInverse(tensor) end
	
	if (method == "LUDecomposition") then return luDecompositionInverse(tensor) end

	error("Invalid method.")

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
	
	local numberOfRows = #tensor

	local numberOfColumns = #tensor[1]

	for rowIndex = 1, numberOfRows, 1 do

		for columnIndex = 1, numberOfColumns, 1 do

			local value = tensor[rowIndex][columnIndex]

			if (value ~= value) then return {rowIndex, columnIndex} end

		end

	end

	return nil

end

function AqwamTensorLibrary:findValue(tensor, valueToFind)
	
	local numberOfRows = #tensor
	
	local numberOfColumns = #tensor[1]

	for rowIndex = 1, numberOfRows, 1 do

		for columnIndex = 1, numberOfColumns, 1 do 

			if (tensor[rowIndex][columnIndex] == valueToFind) then return {rowIndex, columnIndex} end

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
			
			local resultVector = {}
			
			for j = 1, numberOfColumns, 1 do
				
				resultVector[j] = tensor[(numberOfRows - i) + 1][j]
				
			end
			
			resultTensor[i] = resultVector
			
		end
		
	elseif (dimension == 2) then
		
		for i = 1, numberOfRows, 1 do

			local resultVector = {}
			
			for j = 1, numberOfColumns, 1 do
				
				resultVector[j] = tensor[i][(numberOfColumns - j) + 1]
				
			end
			
			resultTensor[i] = resultVector

		end
		
	else
		
		error("Invalid dimension.")
		
	end

	return resultTensor
	
end

function AqwamTensorLibrary:sample(tensor, dimension)
	
	if (dimension <= 0) then error("The dimension cannot be less than or equal to zero.") end
	
	if (dimension > 2) then error("The dimension cannot be greater than 2.") end

	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfRows = dimensionSizeArray[1]

	local numberOfColumns = dimensionSizeArray[2]
	
	local absoluteTensor = AqwamTensorLibrary:applyFunction(math.abs, tensor)

	local sumAbsoluteTensor = AqwamTensorLibrary:sum(absoluteTensor, dimension)

	local probabilityTensor = AqwamTensorLibrary:divide(absoluteTensor, sumAbsoluteTensor)
	
	local newDimensionSizeArray = table.clone(dimensionSizeArray)
	
	newDimensionSizeArray[dimension] = 1

	local randomProbabilityTensor = AqwamTensorLibrary:createRandomUniformTensor(newDimensionSizeArray, 0, 1)

	local resultTensor = {}
	
	if (dimension == 1) then 
		
		tensor = AqwamTensorLibrary:transpose(tensor)
		
		numberOfRows, numberOfColumns = numberOfColumns, numberOfRows
		
	end
	
	for i = 1, numberOfRows, 1 do

		local unwrappedProbabilityVector = probabilityTensor[i]

		local randomProbabilityValue = randomProbabilityTensor[i][1]

		local cumulativeProbabilityValue = 0

		local index

		for j = 1, numberOfColumns, 1 do

			cumulativeProbabilityValue = cumulativeProbabilityValue + unwrappedProbabilityVector[j]

			if (cumulativeProbabilityValue >= randomProbabilityValue) then

				index = j

				break

			end

		end

		resultTensor[i] = {index}

	end
	
	if (dimension == 1) then resultTensor = AqwamTensorLibrary:transpose(resultTensor) end

	return resultTensor

end

function AqwamTensorLibrary:createUpperTriangularTensor(dimensionSizeArray, diagonalValue, offDiagonalValue)

	local numberOfRows = dimensionSizeArray[1]

	local numberOfColumns = dimensionSizeArray[2]

	local resultTensor = {}

	local resultVector

	diagonalValue = diagonalValue or 1

	offDiagonalValue = offDiagonalValue or diagonalValue

	for i = 1, numberOfRows, 1 do

		resultVector = table.create(numberOfColumns, 0)

		for j = i, numberOfColumns, 1 do 

			resultVector[j] = ((j == i) and diagonalValue) or offDiagonalValue

		end

		resultTensor[i] = resultVector

	end

	return resultTensor

end

function AqwamTensorLibrary:createLowerTriangularTensor(dimensionSizeArray, diagonalValue, offDiagonalValue)
	
	local numberOfRows = dimensionSizeArray[1]
	
	local numberOfColumns = dimensionSizeArray[2]
	
	local resultTensor = {}
	
	local resultVector
	
	diagonalValue = diagonalValue or 1
	
	offDiagonalValue = offDiagonalValue or diagonalValue
	
	for i = 1, numberOfRows, 1 do
		
		resultVector = table.create(numberOfColumns, 0)
		
		for j = 1, i, 1 do 
			
			resultVector[j] = ((j == i) and diagonalValue) or offDiagonalValue
			
		end
		
		resultTensor[i] = resultVector
		
	end
	
	return resultTensor
	
end

function AqwamTensorLibrary:convertToDiagonalTensor(tensor)
	
	local dimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)
	
	local numberOfRows = dimensionSizeArray[1]
	
	local numberOfColumns = dimensionSizeArray[2]
	
	if (numberOfRows ~= 1) and (numberOfColumns ~= 1) then error("Invalid tensor.") end
	
	local resultTensor = {}
	
	if (numberOfRows == 1) then
		
		for i = 1, numberOfColumns, 1 do
			
			local resultVector = table.create(numberOfColumns, 0)
			
			resultVector[i] = tensor[1][i]
			
			resultTensor[i] = resultVector
			
		end
		
	elseif (numberOfColumns == 1) then
		
		for i = 1, numberOfRows, 1 do

			local resultVector = table.create(numberOfRows, 0)

			resultVector[i] = tensor[i][1]

			resultTensor[i] = resultVector

		end
		
	end
	
	return resultTensor
	
end

return AqwamTensorLibrary
