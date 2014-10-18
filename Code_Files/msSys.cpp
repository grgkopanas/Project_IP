/*******************************************************

                 Mean Shift Analysis Library
	=============================================

	The mean shift library is a collection of routines
	that use the mean shift algorithm. Using this algorithm,
	the necessary output will be generated needed
	to analyze a given input set of data.

  Mean Shift System:
  ==================

	The Mean Shift System class provides a mechanism for the
	mean shift library classes to prompt progress and to
	time its computations. When porting the mean shift library
	to an application the methods of this class may be changed
	such that the output of the mean shift class prompts
	will be given to whatever hardware or software device that
	is desired.

	The definition for the mean shift system class is provided
	below. Its prototype is provided in "msSys.cc".

The theory is described in the papers:

  D. Comaniciu, P. Meer: Mean Shift: A robust approach toward feature
									 space analysis.

  C. Christoudias, B. Georgescu, P. Meer: Synergism in low level vision.

and they are is available at:
  http://www.caip.rutgers.edu/riul/research/papers/

Implemented by Chris M. Christoudias, Bogdan Georgescu
********************************************************/

//include the msSystem class prototype
#include	"msSys.h"

//include needed system libraries
#include	<time.h>
#include	<stdio.h>
#include	<stdarg.h>
#include	<stdlib.h>

/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      PUBLIC METHODS     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/

  /*/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\*/
  /*** Class Constructor and Destructor ***/
  /*\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/*/

/*******************************************************/
/*Class Constructor                                    */
/*******************************************************/
/*Constructs a mean shift system object.               */
/*******************************************************/
/*Post:                                                */
/*      - an msSystem object has been properly init-   */
/*        ialized.                                     */
/*******************************************************/



/*******************************************************/
/*Class Destructor                                     */
/*******************************************************/
/*Destroys a mean shift system object.                 */
/*******************************************************/
/*Post:                                                */
/*      - an msSystem object has been properly dest-   */
/*        royed.                                       */
/*******************************************************/


 /*/\/\/\/\/\/\/\/\/\*/
 /*** System Timer ***/
 /*\/\/\/\/\/\/\/\/\/*/

/*******************************************************/
/*Start Timer                                          */
/*******************************************************/
/*Sets the mean shift system time to the current       */
/*system time.                                         */
/*******************************************************/
/*Post:                                                */
/*      - the mean shift system time has been set to   */
/*        the current system time.                     */
/*******************************************************/



/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ END OF CLASS DEFINITION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/
/*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*/