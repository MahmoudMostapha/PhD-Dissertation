//
//  myOptions.cpp
//  ParticlesGUI
//
//  Created by Joohwi Lee on 2/11/13.
//
//

#include "piOptions.h"
#include "sstream"
#include "iostream"
#include "fstream"

using namespace std;

namespace pi {
    void Options::SetBool(std::string name, bool value) {
        std::pair<BoolMap::iterator, bool> result = _boolMap.insert(BoolPair(name, value));
        if (!result.second) {
            result.first->second = value;
        }
    }
    
    void Options::SetInt(std::string name, int value) {
        std::pair<IntMap::iterator, bool> result = _intMap.insert(IntPair(name, value));
        if (!result.second) {
            result.first->second = value;
        }
    }

    void Options::SetReal(std::string name, OptionReal value) {
        std::pair<RealMap::iterator, bool> result = _realMap.insert(RealPair(name, value));
        if (!result.second) {
            result.first->second = value;
        }    }

    void Options::SetString(std::string name, std::string value) {
        std::pair<StringMap::iterator, bool> result = _stringMap.insert(StringPair(name, value));
        if (!result.second) {
            result.first->second = value;
        }
    }

    void Options::AppendString(std::string name, std::string value) {
        _stringVectorMap[name].push_back(value);
    }

    void Options::AppendReal(std::string name, OptionReal value) {
        _realVectorMap[name].push_back(value);
    }

    void Options::AppendInt(std::string name, int value) {
        _intVectorMap[name].push_back(value);
    }


    bool Options::GetBoolTo(std::string name, bool& var) {
        if (_boolMap.find(name) == _boolMap.end()) {
            return false;
        }
        var = _boolMap.at(name);
        return true;
    }

    bool Options::GetIntTo(std::string name, int& var) {
        if (_intMap.find(name) == _intMap.end()) {
            return false;
        }
        var = _intMap.at(name);
        return true;
    }

    bool Options::GetRealTo(std::string name, OptionReal& var) {
        if (_realMap.find(name) == _realMap.end()) {
            return false;
        }
        var = _realMap.at(name);
        return true;
    }

    bool Options::GetStringTo(std::string name, std::string& var) {
        if (_stringMap.find(name) == _stringMap.end()) {
            return false;
        }
        var = _stringMap.at(name);
        return true;
    }

    bool Options::GetStringVectorValueTo(std::string name, int n, std::string& out) {
        StringVector& vector = _stringVectorMap[name];
        if (vector.size() > n) {
            out = vector[n];
            return true;
        }
        return false;
    }

    bool Options::GetIntVectorValueTo(std::string name, int n, int& out) {
        IntVector& vector = _intVectorMap[name];
        if (vector.size() > n) {
            out = vector[n];
            return true;
        }
        return false;
    }

    bool Options::GetRealVectorValueTo(std::string name, int n, OptionReal& out) {
        RealVector& vector = _realVectorMap[name];
        if (vector.size() > n) {
            out = vector[n];
            return true;
        }
        return false;
    }

    bool Options::GetBool(std::string name, bool def) {
        if (_boolMap.find(name) == _boolMap.end()) {
            return def;
        } else {
            return _boolMap.at(name);
        }
    }


    int Options::GetInt(std::string name, int def) {
        if (_intMap.find(name) == _intMap.end()) {
            return def;
        } else {
            return _intMap.at(name);
        }
    }

    int Options::GetStringAsInt(std::string name, int def) {
        string val = GetString(name, "");
        if (val != "") {
            return atoi(val.c_str());
        }
        return def;
    }

    OptionReal Options::GetReal(std::string name, OptionReal def) {
        if (_realMap.find(name) == _realMap.end()) {
            return def;
        } else {
            return _realMap.at(name);
        }
    }

    OptionReal Options::GetStringAsReal(std::string name, OptionReal def) {
        string val = GetString(name, "");
        if (val != "") {
            return atof(val.c_str());
        }
        return def;
    }

	bool Options::HasString(std::string name) {
		return _stringMap.find(name) != _stringMap.end();
	}
	
    std::string Options::GetString(std::string name, std::string def) {
        if (_stringMap.find(name) == _stringMap.end()) {
            return def;
        } else {
            return _stringMap.at(name);
        }
    }

    StringVector& Options::GetStringVector(std::string name) {
        return _stringVectorMap[name];
    }

    std::string Options::GetStringVectorValue(std::string name, int n, std::string def) {
        StringVector& vector = _stringVectorMap[name];
        if (vector.size() > n) {
            return vector[n];
        }
        return def;
    }

    RealVector& Options::GetRealVector(std::string name) {
        return _realVectorMap[name];
    }

    OptionReal Options::GetRealVectorValue(std::string name, int nth, OptionReal def) {
        RealVector& vector = _realVectorMap[name];
        if (vector.size() > nth) {
            return vector[nth];
        }
        return def;
    }

    IntVector& Options::GetIntVector(std::string name) {
        return _intVectorMap[name];
    }

    int Options::GetIntVectorValue(std::string name, int nth, int def) {
        IntVector& vector = _intVectorMap[name];
        if (vector.size() > nth) {
            return vector[nth];
        }
        return def;
    }

    std::string Options::GetConfigFile() {
        if (GetString("--config") != "") {
            return GetString("--config");
        } else {
            bool hasConfigFile = false;
            ifstream in("config.txt");
            hasConfigFile = in.is_open();
            in.close();
            if (hasConfigFile) {
                return string("config.txt");
            }
        }
        return "";
    }
    

    StringVector& Options::ParseOptions(int argc, char* argv[], CSimpleOpt::SOption* optionSpecs) {
        const CSimpleOpt::SOption endOption= { -1, NULL, SO_NONE };

        if (optionSpecs == NULL) {
			if (_specs.size() == 0 || _specs.back().nId > -1) {
				_specs.push_back(endOption);
			}
			optionSpecs = &_specs[0];
            for (int i = 0; i < _specs.size(); i++) {
                optionSpecs[i].pszArg = _specNames[i].c_str();
            }
        }

        CSimpleOpt args(argc, argv, optionSpecs);
        while (args.Next()) {
            if (args.LastError() == SO_SUCCESS) {
                if (args.OptionArg()) {
                    this->SetString(args.OptionText(), string(args.OptionArg()));
                } else {
                    this->SetBool(args.OptionText(), true);
                }
            } else {
                // handle error (see the error codes - enum ESOError)
                std::cout << "ERROR:" << args.LastError() << std::endl;
            }
        }

        for (int i = 0; i < args.FileCount(); i++) {
            this->AppendString("args", args.File(i));
        }

        return this->GetStringVector("args");
    }

    std::ostream & operator<<(std::ostream &os, const Options& opt) {
        bool hasValue = false;
        for(Options::BoolMap::const_iterator iter = opt._boolMap.begin(); iter != opt._boolMap.end(); ++iter) {
            if (!hasValue) {
                os << "Options:";
                hasValue = true;
            }
            if (iter->first[0] != '_') {
                os << " " << (iter->second ? "+" : "-") << iter->first;
            }
        }
        if (hasValue) {
            os << endl;
            hasValue = false;
        }

        for(Options::IntMap::const_iterator iter = opt._intMap.begin(); iter != opt._intMap.end(); ++iter) {
            os << iter->first << " int " << iter->second << endl;
        }
        for(Options::RealMap::const_iterator iter = opt._realMap.begin(); iter != opt._realMap.end(); ++iter) {
            os << iter->first << " real " << iter->second << endl;
        }
        for(Options::StringMap::const_iterator iter = opt._stringMap.begin(); iter != opt._stringMap.end(); ++iter) {
            os << iter->first << " string " << iter->second << endl;
        }

        for(Options::IntVectorMap::const_iterator iter = opt._intVectorMap.begin(); iter != opt._intVectorMap.end(); ++iter) {
            IntVector vector = iter->second;
            os << iter->first << " ints";
            for (int i = 0; i < vector.size(); i++) {
                os << " " << vector[i];
            }
            os << endl;
        }

        for(Options::RealVectorMap::const_iterator iter = opt._realVectorMap.begin(); iter != opt._realVectorMap.end(); ++iter) {
            RealVector vector = iter->second;
            os << iter->first << " reals";
            for (int i = 0; i < vector.size(); i++) {
                os << " " << vector[i];
            }
            os << endl;
        }

        for(Options::StringVectorMap::const_iterator iter = opt._stringVectorMap.begin(); iter != opt._stringVectorMap.end(); ++iter) {
            StringVector vector = iter->second;
            os << iter->first << " strings " << vector.size() << endl;
            for (int i = 0; i < vector.size(); i++) {
                os << vector[i] << endl;
            }
        }
        return os;
    }

    std::istream & operator>>(std::istream &is, Options& opt) {
        char *cbuf = new char[10240];
        while (is.good()) {
            is.getline(cbuf, 10240);
            if (is.good()) {
                string line(cbuf);
                stringstream ss(line);
                string name;
                string type;
                ss >> name;
                // skip whitespace
                if (name[0] == '#' || name[0] == '%') {
                    continue;
                }
                if (name == "Options:") {
                    while (ss.good()) {
                        string option;
                        ss >> option;
                        if (option != "") {
                            bool value = (option[0] == '+');
                            string optionname = option.substr(1);
                            opt.SetBool(optionname, value);
                        }
                    }
                }
                ss >> type;
                if (name == OPTION_END) {
                    return is;
                }
                if (type == "int" ) {
                    int value;
                    ss >> value;
                    opt.SetInt(name, value);
                } else if (type == "real") {
                    OptionReal value;
                    ss >> value;
                    opt.SetReal(name, value);
                } else if (type == "bool") {
                    string value;
                    ss >> value;
                    opt.SetBool(name, value == "true");
                } else if (type == "string") {
                    string value;
                    ss >> value;
                    opt.SetString(name, value);
                } else if (type == "ints") {
                    while (ss.good()) {
                        int val;
                        ss >> val;
                        opt.AppendInt(name, val);
                    }
                } else if (type == "reals") {
                    while (ss.good()) {
                        OptionReal val;
                        ss >> val;
                        opt.AppendReal(name, val);
                    }
                } else if (type == "strings") {
                    int len;
                    ss >> len;
                    for (int i = 0; i < len; i++) {
                        char sbuf[1024];
                        is.getline(sbuf, sizeof(sbuf));
                        if (is.good()) {
                            opt.AppendString(name, sbuf);
                        }
                    }
                }
            }
        }
        delete[] cbuf;
        return is;
    }


    StringVector Options::GetSplitString(std::string name, std::string tok, std::string def) {
        std::string val = GetString(name, def);
        return Split(val, tok);
    }

    
    IntVector Options::GetSplitIntVector(std::string name, std::string tok) {
        StringVector data = GetSplitString(name, tok);
        IntVector intVector;
        for (int i = 0; i < data.size(); i++) {
            intVector.push_back(atoi(data[i].c_str()));
        }
        return intVector;
    }


    DoubleVector Options::GetSplitDoubleVector(std::string name, std::string tok) {
        StringVector data = GetSplitString(name, tok);
        DoubleVector doubleVector;
        for (int i = 0; i < data.size(); i++) {
            doubleVector.push_back(atof(data[i].c_str()));
        }
        return doubleVector;
    }

    
    IntVector Options::GetStringAsIntVector(std::string name) {
        StringVector data = GetSplitString(name, ",");
        IntVector intVector;
        for (int i = 0; i < data.size(); i++) {
            intVector.push_back(atoi(data[i].c_str()));
        }
        return intVector;
    }


	void Options::addOption(std::string name, int argType) {
		_specNames.push_back(name);
		CSimpleOpt::SOption opt;
		opt.nId = _specs.size();
		opt.pszArg = _specNames.back().c_str();
		opt.nArgType = (_ESOArgType) argType;
		_specs.push_back(opt);
	}

    void Options::addOption(std::string name, string help, int argType) {
		_specNames.push_back(name);

		CSimpleOpt::SOption opt;
		opt.nId = _specs.size();
		opt.pszArg = _specNames.back().c_str();
		opt.nArgType = (_ESOArgType) argType;
        _specs.push_back(opt);

        std::pair<StringMap::iterator, bool> result = _specHelpMessages.insert(StringPair(name, help));
        if (!result.second) {
            result.first->second = help;
        }
	}

    void Options::addOption(std::string name, string help, string usage, int argType) {
		_specNames.push_back(name);

		CSimpleOpt::SOption opt;
		opt.nId = _specs.size();
		opt.pszArg = _specNames.back().c_str();
		opt.nArgType = (_ESOArgType) argType;
        _specs.push_back(opt);

        std::pair<StringMap::iterator, bool> result = _specHelpMessages.insert(StringPair(name, help));
        if (!result.second) {
            result.first->second = help;
        }

        std::pair<StringMap::iterator, bool> resultUsage = _specUsage.insert(StringPair(name, usage));
        if (!resultUsage.second) {
            resultUsage.first->second = usage;
        }
	}

    std::string Options::GetOptionHelp(std::string name) {
        if (_specHelpMessages.find(name) == _specHelpMessages.end()) {
            return "";
        } else {
            return _specHelpMessages.at(name);
        }
    }

    std::string Options::GetOptionUsage(std::string name) {
        if (_specUsage.find(name) == _specUsage.end()) {
            return "";
        } else {
            return _specUsage.at(name);
        }
    }


    StringVector& Options::GetOptionNames() {
        return _specNames;
    }


    // print usage
    void Options::PrintUsage() {
        StringVector& specs = GetOptionNames();
        for (int i = 0; i < specs.size(); i++) {
            string name = specs[i];
            string help = GetOptionHelp(name);
            string usage = GetOptionUsage(name);

            cout << "* " << name << endl;
            if (help != "") {
                cout << "\t* " << help << endl;
            }
            if (usage != "") {
                cout << "\t* *ex)* " << usage << endl;
            }
        }
    }

    void Options::main(Options& opts, StringVector& args) {
    }

    //////////////////////////////////////////////////////////////////////////////////////
    //
    // STATIC FUNCTIONS
    //
    //////////////////////////////////////////////////////////////////////////////////////
    StringVector Options::Split(std::string str, std::string tok) {
        StringVector data;
        istringstream is(str);
        string part;
        while (getline(is, part, tok[0])) {
            data.push_back(part);
        }
        return data;
    }
    
    
    IntVector Options::SplitAsInt(std::string str, char tok) {
        IntVector data;
        istringstream is(str);
        string part;
        while (getline(is, part, tok)) {
            data.push_back(atoi(part.c_str()));
        }
        return data;
    }

    DoubleVector Options::SplitAsDouble(std::string str, char tok) {
        DoubleVector data;
        istringstream is(str);
        string part;
        while (getline(is, part, tok)) {
            data.push_back(atof(part.c_str()));
        }
        return data;
    }
    
}

