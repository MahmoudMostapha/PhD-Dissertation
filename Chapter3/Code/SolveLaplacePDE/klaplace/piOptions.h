//
//  myOptions.h
//  ParticlesGUI
//
//  Created by Joohwi Lee on 2/11/13.
//
//

#ifndef __ParticlesGUI__myOptions__
#define __ParticlesGUI__myOptions__

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "SimpleOpt.h"


/**
 * Todo
 * 
 *   value interpolated string 
 *      example)
 *         NumberOfParticles: int 100
 *         Name: interpolated_string file_#NumberOfParticles#.txt
 *         GetString("Name", "") => file_100.txt
 *
 *    if the name of bool type option startswith '_', then it will not be written
 *      example)
 *         Options: +_load_position_only +hello
 *      will be written as
 *         Options: +hello
 *
 */
#define OPTION_END "==option_end=="
namespace pi {
    typedef float OptionReal;
    typedef std::vector<std::string> StringVector;
    typedef std::vector<OptionReal> RealVector;
    typedef std::vector<int> IntVector;
    typedef std::vector<char> CharVector;
    typedef std::vector<double> DoubleVector;
	typedef std::vector<CSimpleOpt::SOption> OptionSpecs;
    
    class Options {
    private:
        typedef std::map<std::string, std::string> StringMap;
        typedef std::pair<std::string, std::string> StringPair;
        typedef std::map<std::string, int> IntMap;
        typedef std::pair<std::string, int> IntPair;
        typedef std::map<std::string, OptionReal> RealMap;
        typedef std::pair<std::string, OptionReal> RealPair;
        typedef std::map<std::string, bool> BoolMap;
        typedef std::pair<std::string, bool> BoolPair;
        typedef std::map<std::string, StringVector> StringVectorMap;
        typedef std::pair<std::string, StringVector> StringVectorPair;
        typedef std::map<std::string, RealVector> RealVectorMap;
        typedef std::pair<std::string, RealVector> RealVectorPair;
        typedef std::map<std::string, IntVector> IntVectorMap;
        typedef std::pair<std::string, IntVector> IntVectorPair;
    public:
        void SetBool(std::string name, bool value);
        void SetInt(std::string name, int value);
        void SetReal(std::string name, OptionReal value);
        void SetString(std::string name, std::string value);
        void AppendString(std::string name, std::string value);
        void AppendReal(std::string name, OptionReal value);
        void AppendInt(std::string name, int value);

        bool GetBoolTo(std::string name, bool& var);
        bool GetIntTo(std::string name, int& var);
        bool GetRealTo(std::string name, OptionReal& var);
        bool GetStringTo(std::string name, std::string& var);
        bool GetRealVectorValueTo(std::string name, int n, OptionReal& out);
        bool GetIntVectorValueTo(std::string name, int n, int& out);
        bool GetStringVectorValueTo(std::string name, int n, std::string& out);

        bool GetBool(std::string name, bool def = false);
        int GetInt(std::string name, int def);
        int GetStringAsInt(std::string name, int def);
        OptionReal GetReal(std::string name, OptionReal def);
        OptionReal GetStringAsReal(std::string name, OptionReal def);
        std::string GetString(std::string name, std::string def = "");
		bool HasString(std::string name);
		
        StringVector& GetStringVector(std::string name);
        std::string GetStringVectorValue(std::string name, int i, std::string def = "");
        StringVector GetSplitString(std::string name, std::string tok, std::string def = "");
        IntVector GetSplitIntVector(std::string name, std::string tok);
        DoubleVector GetSplitDoubleVector(std::string name, std::string tok);
        
        IntVector GetStringAsIntVector(std::string name);        
        
        RealVector& GetRealVector(std::string name);
        OptionReal GetRealVectorValue(std::string name, int nth, OptionReal def = 0);

        IntVector& GetIntVector(std::string name);
        int GetIntVectorValue(std::string name, int nth, int def = 0);


        std::string GetConfigFile();

        StringVector& ParseOptions(int argc, char* argv[], CSimpleOpt::SOption*);
        static StringVector Split(std::string str, std::string tok);
        static IntVector SplitAsInt(std::string str, char tok);
        static DoubleVector SplitAsDouble(std::string str, char tok);
        

		void addOption(std::string name, int argType);
		void addOption(std::string name, std::string help, int argType);
		void addOption(std::string name, std::string help, std::string usage, int argType);

        /// return a help message for an option name
        /// \param name option name
        /// \return help message for a given option name
        std::string GetOptionHelp(std::string name);


        /// return a usage for an option name
        /// \param name option name
        /// \return usage for a given option name
        std::string GetOptionUsage(std::string name);


        /// return specNames
        /// \return a StringVector contains all option names
        StringVector& GetOptionNames();

        /// print usage
        void PrintUsage();


        /// main function for test
        void main(Options& opts, StringVector& args);

    private:
        StringVector _specNames;
		OptionSpecs _specs;
        StringMap _specHelpMessages;
        StringMap _specUsage;

        BoolMap _boolMap;
        StringMap _stringMap;
        IntMap _intMap;
        RealMap _realMap;
        StringVectorMap _stringVectorMap;
        RealVectorMap _realVectorMap;
        IntVectorMap _intVectorMap;

        friend std::ostream & operator<<(std::ostream &os, const Options& opt);
        friend std::istream & operator>>(std::istream &is, Options& opt);
    };
}
#endif /* defined(__ParticlesGUI__myOptions__) */
