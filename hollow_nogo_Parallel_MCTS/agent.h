/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
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
#include <fstream>

#include <bits/stdc++.h>
#include <omp.h>

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
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
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
		else
			engine.seed((unsigned)time(NULL));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=random role=unknown N=0 " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
	}

	virtual action take_action(const board& state) {
		int N = meta["N"];
		if(N){
			// root parallelizing
			int thread_num = omp_get_num_procs();
			omp_set_num_threads(thread_num);
			std::vector<int> majority_vote(thread_num, 0);

			//std::fstream debug("record.txt", std::ios::app);

			#pragma omp parallel
			{
				int id = omp_get_thread_num();
				node* root = new node(state);
				int vote_move = root->MCTS(N, engine);
				delete_tree(root);
				majority_vote[id] = vote_move;
			}

			std::vector<int> vote_result(81, 0);
			for(auto &v : majority_vote){
				if(v != -1){
					//debug << v << " ";
					vote_result[v]++;
				}
			}

			//debug << std::endl;
			//debug.close();

			std::vector<int>::iterator iter = max_element(vote_result.begin(), vote_result.end());
			if((*iter) == 0){
				return action();
			}

			int result = iter - vote_result.begin();
			return action::place(result, state.info().who_take_turns);
		}

		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
		}
		return action();
	}

	class node : board {
	public:
		int win_cnt;
		int total_cnt;
		int place_pos;
		std::vector<node*> child;
		node* parent;

		node(const board& state, int m = -1): board(state), place_pos(m), win_cnt(0), total_cnt(0), parent(nullptr) {}

		float win_rate(){
			if(win_cnt == 0 && total_cnt == 0){
				return 0.0;
			}
			
			return (float)win_cnt / total_cnt;
		}

		float ucb(){
			float c = 1.5;

			if(parent->total_cnt == 0 || total_cnt == 0){
				return win_rate();
			}
			
			return win_rate() + c * std::sqrt(std::log(parent->total_cnt) / total_cnt);
		}

		float ucb_opponent(){
			float c = 1.5;

			if(parent->total_cnt == 0 || total_cnt == 0){
				return 1 - win_rate();
			}

			return (1 - win_rate()) + c * std::sqrt(std::log(parent->total_cnt) / total_cnt);
		}

		int MCTS(int N, std::default_random_engine& engine){
			// 1. select  2. expand  3. simulate  4. back propagate

			// debug
			//std::fstream debug("record.txt", std::ios::app);
			
			for(int i = 0; i < N; ++i){
				// select
				//debug << "select" << std::endl;
				std::vector<node*> path = select_root_to_leaf(info().who_take_turns);
				// expand
				//debug << "expand" << std::endl;
				node* leaf = path.back();
				node* expand_node = leaf->expand_from_leaf(engine);
				if(expand_node != leaf){
					path.push_back(expand_node);
				}
				// simulate
				//debug << "simulate" << std::endl;
				unsigned winner = path.back()->simulate_winner(engine);
				// backpropagate
				//debug << "backpropagate" << std::endl;
				back_propagate(path, winner);
			}

			//debug.close();
			return select_action();
		}

		int select_action(){
			// select child node who has the highest win rate (highest Q)
			if(child.size() == 0){
				return -1;
			}

			float max_score = -std::numeric_limits<float>::max();
			node* c;
			for(int i = 0; i < child.size(); ++i){
				float tmp = child[i]->win_rate();
				if(tmp > max_score){
					max_score = tmp;
					c = child[i];
				}
			}
			
			return c->place_pos;
		}

		std::vector<node*> select_root_to_leaf(unsigned who){
			std::vector<node*> vec;
			node* curr = this;
			vec.push_back(curr);
			while(!curr->is_leaf()){
				// select node who has the highest ucb score
				float max_score = -std::numeric_limits<float>::max();
				node* c;
				if(curr->child.size() == 0){
					break;
				}
				for(int i = 0; i < curr->child.size(); ++i){
					float tmp;
					if(who == curr->info().who_take_turns){
						tmp = curr->child[i]->ucb();
					}else{
						tmp = curr->child[i]->ucb_opponent();
					}
					
					if(tmp > max_score){
						max_score = tmp;
						c = curr->child[i];
					}
				}
				vec.push_back(c);
				curr = c;
			}

			return vec;
		}

		bool is_leaf(){
			int cnt = 0;
			for(int i = 0; i < 81; i++){
				if(board(*this).place(i) == board::legal){
					cnt++;
				}
			}
			// check if fully expanded (leaf == not fully expanded)
			return !(cnt > 0 && child.size() == cnt);
		}

		node* expand_from_leaf(std::default_random_engine& engine){
			board b = *this;
			std::vector<int> vec = all_space(engine);
			bool success_placed = 0;
			int pos = -1;
			for(int i = 0; i < vec.size(); ++i){
				if(b.place(vec[i]) == board::legal){
					pos = vec[i];
					success_placed = 1;
					break;
				}
			}

			if(success_placed){
				node* new_node = new node(b, pos);
				this->child.push_back(new_node);
				new_node->parent = this;
				return new_node;
			}else{
				return this;
			}
		}

		unsigned simulate_winner(std::default_random_engine& engine){
			board b = *this;
			std::vector<int> vec = all_space(engine);
			std::queue<int> q;
			for(int i = 0; i < vec.size(); ++i){
				q.push(vec[i]);
			}

			int cnt = 0;
			while(cnt != q.size()){
				int i = q.front();
				q.pop();
				if(b.place(i) != board::legal){
					q.push(i);
					cnt++;
				}else{
					cnt = 0;
				}
			}

			if(b.info().who_take_turns == board::white){
				return board::black;
			}else{
				return board::white;
			}
		}

		std::vector<int> all_space(std::default_random_engine& engine){
			std::vector<int> vec;
			for(int i = 0; i < 81; ++i){
				vec.push_back(i);
			}
			std::shuffle(vec.begin(), vec.end(), engine);
			return vec;
		}

		void back_propagate(std::vector<node*>& path, unsigned winner){
			for(int i = 0; i < path.size(); ++i){
				path[i]->total_cnt++;
				if(winner == info().who_take_turns){
					path[i]->win_cnt++;
				}
			}
		}
	};

	void delete_tree(node* root){
		if(root->child.size() == 0){
			delete root;
			return ;
		}

		for(int i = 0; i < root->child.size(); ++i){
			if(root->child[i] != nullptr){
				delete_tree(root->child[i]);
			}
		}

		delete root;
	}

private:
	std::vector<action::place> space;
	board::piece_type who;
};
