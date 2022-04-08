#pragma once

#include "AI_Variables.h"

#include "AI_Utility.h"

#include "AI_Instance.h"

#include "AI_Functionality.h"

#include "AI_Manager.h"

#include "Resource_Manager.h"


/*

Training data format :
	Each Line is terminated by a Excalamation mark (!)
	First Line:			Input size & output size seperated by a colon (:)
	Following lines :	Data
						Data is seperated by a comma(,)
						At the end of each line is the expected result
						It is seperated from the data by a dash(-)

*/


