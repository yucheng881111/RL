/**
 * Framework for 2048 & 2048-like Games (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Theory of Computer Games (TCG 2021)
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include "weight.h"
#include <fstream>
#include <unistd.h>

struct state{
	board board_before;
	board board_after;
	int reward;
	float value;
	state(){
		reward = 0;
		value = 0;
	}
};

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag, std::vector<state> &vec) {}
	virtual action take_action(const board& b, float& vs, int& r) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class player : public agent {
public:
	player(const std::string& args = "") : agent("name=dummy role=player " + args), alpha(0) {
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~player() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}
	virtual action take_action(const board& before, float& vs, int& r) {
		// TODO: perform expectimax
		
		float val_max = -std::numeric_limits<float>::max();
		int r_max = -2147483648;
		int op = -1;
		for(int i = 0; i < 4; ++i){
			board b = board(before);
			int reward = b.slide(i);
			if(reward == -1){
				continue;
			}

			//float v = reward + estimate_value(b);
			float v = reward + expectation(b);
			if(v > val_max){
				val_max = v;
				r_max = reward;
				op = i;
			}
		}
		vs = val_max;
		r = r_max;
		return action::slide(op);
	}

	virtual void open_episode(const std::string& flag = "") {
		// TODO
	}
	//td
	virtual void close_episode(const std::string& flag, std::vector<state> &path) {
		// TODO
		/*
		float tmp = 0;
		for(int i = path.size() - 1; i >= 0; i--){
			float td_error = tmp - (path[i].value - path[i].reward);
			tmp = path[i].reward + adjust_value(path[i].board_after, alpha * td_error);
		}
		*/
	}
	
	// TODO? = change by yourself if you need

	int extract_feature(const board& after, int a, int b, int c, int d) const {
		return after(a) * 16 * 16 * 16 + after(b) * 16 * 16 + after(c) * 16 + after(d);
	}
	int extract_feature5(const board& after, int a, int b, int c, int d, int e) const {
		// TODO?
		return 0;
	}
	int extract_feature6(const board& after, int a, int b, int c, int d, int e, int f) const {
		// TODO?
		int index = after(a) * 16 * 16 * 16 * 16 * 16 
				  + after(b) * 16 * 16 * 16 * 16
				  + after(c) * 16 * 16 * 16
				  + after(d) * 16 * 16
				  + after(e) * 16
				  + after(f);

		return index;
	}


	float estimate_value(const board& after) const {
		// TODO
		float sum = 0.0;
		board b = board(after);
		
		for(int i = 0; i < 4; ++i){
			
			int idx0 = extract_feature6(b, 0, 1, 2, 3, 4, 5);
			int idx1 = extract_feature6(b, 4, 5, 6, 7, 8, 9);
			int idx2 = extract_feature6(b, 0, 1, 2, 4, 5, 6);
			int idx3 = extract_feature6(b, 4, 5, 6, 8, 9, 10);
			
			/*
			int idx0 = extract_feature(b, 0, 1, 2, 3);
			int idx1 = extract_feature(b, 4, 5, 6, 7);
			int idx2 = extract_feature(b, 8, 9,10,11);
			int idx3 = extract_feature(b,12,13,14,15);
			*/
			sum += net[0][idx0] + net[1][idx1] + net[2][idx2] + net[3][idx3];
			b.rotate_right();
		}
		b.reflect_horizontal();
		
		for(int i = 0; i < 4; ++i){
			
			int idx0 = extract_feature6(b, 0, 1, 2, 3, 4, 5);
			int idx1 = extract_feature6(b, 4, 5, 6, 7, 8, 9);
			int idx2 = extract_feature6(b, 0, 1, 2, 4, 5, 6);
			int idx3 = extract_feature6(b, 4, 5, 6, 8, 9, 10);
			
			/*
			int idx0 = extract_feature(b, 0, 1, 2, 3);
			int idx1 = extract_feature(b, 4, 5, 6, 7);
			int idx2 = extract_feature(b, 8, 9,10,11);
			int idx3 = extract_feature(b,12,13,14,15);
			*/
			sum += net[0][idx0] + net[1][idx1] + net[2][idx2] + net[3][idx3];
			b.rotate_right();
		}
		
		return sum;
	}

	float expectation(const board& after) const {
		float result = 0.0;
		int empty_space = 0;
		for(int pos = 0; pos < 16; ++pos){
			if (after(pos) == 0) empty_space++;
		}

		for(int pos = 0; pos < 16; ++pos){
			if (after(pos) != 0) continue;
			//board::cell tile = popup(engine) ? 1 : 2;
			board b1 = board(after);
			board b2 = board(after);

			b1.place(pos, 1); // place 2
			float val_max1 = -std::numeric_limits<float>::max();
			for(int i = 0; i < 4; ++i){
				board b1_tmp = board(b1);
				int reward = b1_tmp.slide(i);
				if(reward == -1){
					continue;
				}
				float v = reward + estimate_value(b1_tmp);
				val_max1 = std::max(val_max1, v);
			}

			b2.place(pos, 2); // place 4
			float val_max2 = -std::numeric_limits<float>::max();
			for(int i = 0; i < 4; ++i){
				board b2_tmp = board(b2);
				int reward = b2_tmp.slide(i);
				if(reward == -1){
					continue;
				}
				float v = reward + estimate_value(b2_tmp);
				val_max2 = std::max(val_max2, v);
			}

			result += ((val_max1 * 0.9 + val_max2 * 0.1) / empty_space);
		}

		return result;
	}

	
	float adjust_value(const board& after, float target){
		// TODO	
		float u_split = target / 32;
		float sum = 0.0;

		board b = board(after);
		for(int i = 0; i < 4; ++i){
			
			int idx0 = extract_feature6(b, 0, 1, 2, 3, 4, 5);
			int idx1 = extract_feature6(b, 4, 5, 6, 7, 8, 9);
			int idx2 = extract_feature6(b, 0, 1, 2, 4, 5, 6);
			int idx3 = extract_feature6(b, 4, 5, 6, 8, 9, 10);
			
			/*
			int idx0 = extract_feature(b, 0, 1, 2, 3);
			int idx1 = extract_feature(b, 4, 5, 6, 7);
			int idx2 = extract_feature(b, 8, 9,10,11);
			int idx3 = extract_feature(b,12,13,14,15);
			*/
			net[0][idx0] += (u_split);
			net[1][idx1] += (u_split);
			net[2][idx2] += (u_split);
			net[3][idx3] += (u_split);
			sum += net[0][idx0] + net[1][idx1] + net[2][idx2] + net[3][idx3];
			b.rotate_right();
		}
		b.reflect_horizontal();
		for(int i = 0; i < 4; ++i){
			
			int idx0 = extract_feature6(b, 0, 1, 2, 3, 4, 5);
			int idx1 = extract_feature6(b, 4, 5, 6, 7, 8, 9);
			int idx2 = extract_feature6(b, 0, 1, 2, 4, 5, 6);
			int idx3 = extract_feature6(b, 4, 5, 6, 8, 9, 10);
			
			/*
			int idx0 = extract_feature(b, 0, 1, 2, 3);
			int idx1 = extract_feature(b, 4, 5, 6, 7);
			int idx2 = extract_feature(b, 8, 9,10,11);
			int idx3 = extract_feature(b,12,13,14,15);
			*/
			net[0][idx0] += (u_split);
			net[1][idx1] += (u_split);
			net[2][idx2] += (u_split);
			net[3][idx3] += (u_split);
			sum += net[0][idx0] + net[1][idx1] + net[2][idx2] + net[3][idx3];
			b.rotate_right();
		}

		return sum;
	}

protected:
	virtual void init_weights(const std::string& info) {
//		net.emplace_back(65536); // create an empty weight table with size 65536
//		net.emplace_back(65536); // create an empty weight table with size 65536
		// TODO?
		
		net.emplace_back(16 * 16 * 16 * 16 * 16 * 16);
		net.emplace_back(16 * 16 * 16 * 16 * 16 * 16);
		net.emplace_back(16 * 16 * 16 * 16 * 16 * 16);
		net.emplace_back(16 * 16 * 16 * 16 * 16 * 16);
		/*
		net.emplace_back(16 * 16 * 16 * 16);
		net.emplace_back(16 * 16 * 16 * 16);
		net.emplace_back(16 * 16 * 16 * 16);
		net.emplace_back(16 * 16 * 16 * 16);
		*/
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	float alpha;
};

/**
 * random environment
 * add a new random tile to an empty cell
 * 2-tile: 90%
 * 4-tile: 10%
 */
class rndenv : public random_agent {
public:
	rndenv(const std::string& args = "") : random_agent("name=random role=environment " + args),
		space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }), popup(0, 9) {}

	virtual action take_action(const board& after, float& vs, int& r) {
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;
			board::cell tile = popup(engine) ? 1 : 2; // true: 1, 2, 3, 4, 5, 6, 7, 8, 9; false: 0
			return action::place(pos, tile);
		}
		return action();
	}

private:
	std::array<int, 16> space;
	std::uniform_int_distribution<int> popup;
};

/**
 * dummy player
 * select a legal action randomly
 */
class dummy_player : public random_agent {
public:
	dummy_player(const std::string& args = "") : random_agent("name=dummy role=player " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before, float& vs, int& r) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};
