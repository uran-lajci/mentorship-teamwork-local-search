// Local search optimizer for Hash Code 2022 "Mentorship and Teamwork".
//
// Usage:   ./optimizer <input> <current_solution> <output>
// Time limit is hard-coded to ~5 minutes (see TIME_LIMIT_SEC below).
//
// Approach:
//   - Parse the input and the given solution.
//   - Build a simulator that takes an ordered list of "assignments"
//     (project id + contributor id per role), runs them in order just
//     like the judge does, and returns a score (or -1 if invalid).
//     A greedy "earliest start" rule is used: each project starts on
//     max(av[c]) over its contributors; mentorship is checked with the
//     skills the contributors have *at that moment*.
//   - Local search with several neighborhood moves:
//         * swap two contributors inside the same project
//         * replace one contributor in a project with an unused one
//         * swap a contributor between two different assigned projects
//         * swap the order of two assigned projects
//         * drop an assigned project
//         * try to insert one of the non-assigned projects at the end
//   - Simulated annealing acceptance (very light).
//
// The simulator is written to be as cheap as possible: O(sum |roles|)
// per evaluation, which is small because the used solution is always
// a subset of the 1000 projects.

#include <bits/stdc++.h>
using namespace std;

static const double TIME_LIMIT_SEC = 300.0;     // 5 minutes

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------
struct Project {
    string name;
    int D, S, B;                       // duration, score, best_before
    vector<pair<int,int>> req;         // (skill_id, level) per role
};

struct Contributor {
    string name;
    vector<int> skill;                 // skill[sid] = level, 0 if absent
};

int C_, P_, NSK;
vector<Contributor> CON;
vector<Project>     PRJ;
unordered_map<string,int> skill2id;
unordered_map<string,int> c2i, p2i;

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------
static int getSkillId(const string& s){
    auto it = skill2id.find(s);
    if (it != skill2id.end()) return it->second;
    int id = skill2id.size();
    skill2id[s] = id;
    return id;
}

static void readInput(const string& path){
    ifstream in(path);
    if(!in){ cerr<<"cannot open "<<path<<"\n"; exit(1); }
    in >> C_ >> P_;
    CON.resize(C_);
    for (int c = 0; c < C_; ++c){
        int n; in >> CON[c].name >> n;
        while (n--){
            string s; int l; in >> s >> l;
            int sid = getSkillId(s);
            if ((int)CON[c].skill.size() <= sid) CON[c].skill.resize(sid+1, 0);
            CON[c].skill[sid] = l;
        }
        c2i[CON[c].name] = c;
    }
    PRJ.resize(P_);
    for (int p = 0; p < P_; ++p){
        int r; in >> PRJ[p].name >> PRJ[p].D >> PRJ[p].S >> PRJ[p].B >> r;
        PRJ[p].req.resize(r);
        for (int i = 0; i < r; ++i){
            string s; int l; in >> s >> l;
            int sid = getSkillId(s);
            PRJ[p].req[i] = {sid, l};
        }
        p2i[PRJ[p].name] = p;
    }
    NSK = skill2id.size();
    // Normalise skill vectors to full length.
    for (auto& ct : CON) ct.skill.resize(NSK, 0);
}

// ---------------------------------------------------------------------------
// Solution representation
// ---------------------------------------------------------------------------
// An assignment = (project id, [contributor ids in role order])
using Assign = pair<int, vector<int>>;
using Solution = vector<Assign>;

static Solution readSolution(const string& path){
    ifstream in(path);
    if(!in){ cerr<<"cannot open "<<path<<"\n"; exit(1); }
    int E; in >> E;
    Solution sol; sol.reserve(E);
    in.ignore();
    for (int i = 0; i < E; ++i){
        string projName;
        // skip possible empty lines
        while (getline(in, projName) && projName.empty()) {}
        // Trim trailing CR/spaces
        while (!projName.empty() && (projName.back()=='\r'||projName.back()==' '))
            projName.pop_back();
        auto pit = p2i.find(projName);
        if (pit == p2i.end()){
            cerr<<"unknown project in solution: '"<<projName<<"'\n"; exit(1);
        }
        int pid = pit->second;
        string line;
        if (!getline(in, line)){ cerr<<"truncated solution\n"; exit(1); }
        istringstream iss(line);
        vector<int> cs;
        string nm;
        while (iss >> nm){
            auto cit = c2i.find(nm);
            if (cit == c2i.end()){
                cerr<<"unknown contributor: '"<<nm<<"'\n"; exit(1);
            }
            cs.push_back(cit->second);
        }
        sol.emplace_back(pid, std::move(cs));
    }
    return sol;
}

// ---------------------------------------------------------------------------
// Simulator - evaluate a solution, returning total score and per-project
// information. Returns -1 if any project is invalid.
// ---------------------------------------------------------------------------
struct SimResult {
    long long score = 0;
    bool valid = true;
    // For each assignment index: finish day, and score obtained.
    vector<int> endDay;
    vector<int> projScore;
};

// Working buffers (reused across evaluations).
static vector<int>          g_av;          // av[c]
static vector<vector<int>>  g_skill;       // g_skill[c][s]
static vector<int>          g_teamMax;     // size NSK
static vector<char>         g_usedInProj;  // C_

static void ensureBuffers(){
    if ((int)g_av.size() != C_){
        g_av.assign(C_, 0);
        g_skill.assign(C_, vector<int>());
        for (int c = 0; c < C_; ++c) g_skill[c] = CON[c].skill;
        g_teamMax.assign(NSK, 0);
        g_usedInProj.assign(C_, 0);
    }
}

static SimResult simulate(const Solution& sol){
    SimResult R;
    R.endDay.resize(sol.size());
    R.projScore.resize(sol.size());

    // Reset working state.
    for (int c = 0; c < C_; ++c){
        g_av[c] = 0;
        // copy original skills
        memcpy(g_skill[c].data(), CON[c].skill.data(), NSK*sizeof(int));
    }

    for (size_t k = 0; k < sol.size(); ++k){
        int p = sol[k].first;
        const auto& cs = sol[k].second;
        const auto& req = PRJ[p].req;
        if (cs.size() != req.size()){ R.valid = false; return R; }

        // start day = max availability
        int start = 0;
        for (int c : cs){
            if (c < 0 || c >= C_){ R.valid = false; return R; }
            if (g_av[c] > start) start = g_av[c];
        }

        // check "uses each contributor only once" inside the project
        for (int c : cs){
            if (g_usedInProj[c]){
                // cleanup then fail
                for (int cc : cs) g_usedInProj[cc] = 0;
                R.valid = false; return R;
            }
            g_usedInProj[c] = 1;
        }
        for (int c : cs) g_usedInProj[c] = 0;

        // team skill max
        // Only touch the skills that matter + the skills of the team.
        // Simpler: loop all required skills and also record participants.
        // Mentorship check: for each role (s,l), skill[c][s] >= l OR
        // (skill[c][s] == l-1 AND some other member has skill >= l).
        // Build max skill in team for each required skill.
        // Use small map since req sizes are modest per project.
        static vector<int> teamMaxForReq;
        teamMaxForReq.assign(req.size(), 0);
        // For each role's skill, compute team max.
        // O(|req|^2 * |members|) worst case is fine since roles<=100, members<=100.
        int R_ = (int)req.size();
        for (int i = 0; i < R_; ++i){
            int s = req[i].first;
            int mx = 0;
            for (int m = 0; m < R_; ++m){
                int sv = g_skill[cs[m]][s];
                if (sv > mx) mx = sv;
            }
            teamMaxForReq[i] = mx;
        }

        // Validate roles.
        for (int i = 0; i < R_; ++i){
            int s = req[i].first, l = req[i].second;
            int cv = g_skill[cs[i]][s];
            if (cv >= l) continue;
            if (cv == l-1 && teamMaxForReq[i] >= l) continue; // mentored
            R.valid = false; return R;
        }

        int end = start + PRJ[p].D;             // last day of work = end-1, complete on 'end'
        int lateDay = end;                      // "project's last day of work" = end-1;
        // the rule: "if the project last day of work is strictly before the indicated day,
        //  it earns full score". Last day of work = start + D - 1.
        // Late penalty: one less point for each day late.
        int lastWork = end - 1;
        int gained;
        if (lastWork < PRJ[p].B) gained = PRJ[p].S;
        else gained = max(0, PRJ[p].S - (lastWork - PRJ[p].B + 1));

        R.score += gained;
        R.endDay[k] = end;
        R.projScore[k] = gained;

        // Level-up: contributors whose role required skill >= their current level
        // improve by 1.
        for (int i = 0; i < R_; ++i){
            int s = req[i].first, l = req[i].second;
            int c = cs[i];
            if (g_skill[c][s] <= l){
                ++g_skill[c][s];
            }
            g_av[c] = end;
        }
    }
    R.valid = true;
    return R;
}

// ---------------------------------------------------------------------------
// Helpers used by the neighborhood moves
// ---------------------------------------------------------------------------
static double now_sec(){
    using namespace chrono;
    static auto t0 = steady_clock::now();
    return duration_cast<duration<double>>(steady_clock::now() - t0).count();
}

// Pick the cheapest contributor not in `usedMask` that satisfies skill s at
// level >= need (or == need-1 if mentorable). `usedMask` is a bitset vector.
// Not used in critical path — we just try random contributors.

// ---------------------------------------------------------------------------
// Main local search
// ---------------------------------------------------------------------------
int main(int argc, char** argv){
    if (argc < 4){
        cerr << "usage: " << argv[0] << " <input> <solution> <output>\n";
        return 1;
    }
    readInput(argv[1]);
    Solution cur = readSolution(argv[2]);
    ensureBuffers();

    SimResult sr = simulate(cur);
    if (!sr.valid){
        cerr << "initial solution is INVALID\n";
        return 1;
    }
    cerr << "initial score: " << sr.score
         << "   assignments: " << cur.size() << "\n";

    Solution best = cur;
    long long bestScore = sr.score;

    // Pre-compute for each project the set of candidate contributors per role
    // skill. This is done lazily on demand.
    // For each skill id, list contributors by *original* level (descending).
    vector<vector<int>> contribBySkill(NSK);
    for (int s = 0; s < NSK; ++s){
        for (int c = 0; c < C_; ++c)
            if (CON[c].skill[s] > 0) contribBySkill[s].push_back(c);
        sort(contribBySkill[s].begin(), contribBySkill[s].end(),
             [&](int a, int b){ return CON[a].skill[s] > CON[b].skill[s]; });
    }

    // Which projects are currently in the solution?
    vector<char> inSol(P_, 0);
    for (auto& a : cur) inSol[a.first] = 1;

    mt19937_64 rng(12345);

    auto assignedIdx = [&](int pid){
        for (size_t i = 0; i < cur.size(); ++i)
            if (cur[i].first == pid) return (int)i;
        return -1;
    };

    double startT = now_sec();
    long long iters = 0, improves = 0;
    double temperature = 1.0;      // very light SA

    // Periodic report
    double nextReport = startT + 10.0;

    while (now_sec() - startT < TIME_LIMIT_SEC){
        ++iters;
        if (cur.empty()) break;

        int moveType = rng() % 100;
        Solution cand = cur;    // We will modify and re-simulate. Keep it
                                // lightweight — most moves change a handful
                                // of entries; simulate is O(total roles) which
                                // is acceptable.

        bool mutated = false;

        if (moveType < 20){
            // ---- swap two contributors inside the same project ----
            int idx = rng() % cand.size();
            auto& cs = cand[idx].second;
            if (cs.size() >= 2){
                int i = rng() % cs.size();
                int j = rng() % cs.size();
                if (i != j){ swap(cs[i], cs[j]); mutated = true; }
            }
        } else if (moveType < 55){
            // ---- replace one contributor in a project ----
            int idx = rng() % cand.size();
            auto& cs = cand[idx].second;
            int role = rng() % cs.size();
            int pid = cand[idx].first;
            auto [s, l] = PRJ[pid].req[role];
            // Build the set of contributors currently used in this assignment.
            // Candidates: contributors with skill >= l-1 that aren't used here.
            // Use a pool from contribBySkill[s] and also occasionally take a
            // random contributor (lets level-ups propagate).
            const auto& pool = contribBySkill[s];
            if (!pool.empty()){
                int tries = 20;
                while (tries-- > 0){
                    int pick;
                    if (rng()%4 == 0) pick = rng() % C_;
                    else pick = pool[rng() % pool.size()];
                    bool clash = false;
                    for (int c : cs) if (c == pick){ clash = true; break; }
                    if (clash) continue;
                    cs[role] = pick;
                    mutated = true;
                    break;
                }
            }
        } else if (moveType < 70){
            // ---- swap contributors between two different projects ----
            if (cand.size() >= 2){
                int a = rng() % cand.size();
                int b = rng() % cand.size();
                if (a != b){
                    auto& csA = cand[a].second;
                    auto& csB = cand[b].second;
                    int ia = rng() % csA.size();
                    int ib = rng() % csB.size();
                    // Make sure no duplication appears after swap.
                    int ca = csA[ia], cb = csB[ib];
                    bool ok = true;
                    for (size_t k = 0; k < csA.size(); ++k)
                        if ((int)k != ia && csA[k] == cb){ ok=false; break; }
                    if (ok) for (size_t k = 0; k < csB.size(); ++k)
                        if ((int)k != ib && csB[k] == ca){ ok=false; break; }
                    if (ok){ swap(csA[ia], csB[ib]); mutated = true; }
                }
            }
        } else if (moveType < 80){
            // ---- swap the order of two adjacent assigned projects ----
            if (cand.size() >= 2){
                int a = rng() % (cand.size()-1);
                swap(cand[a], cand[a+1]);
                mutated = true;
            }
        } else if (moveType < 85){
            // ---- move a project to a new random position ----
            if (cand.size() >= 2){
                int a = rng() % cand.size();
                int b = rng() % cand.size();
                if (a != b){
                    auto tmp = cand[a];
                    cand.erase(cand.begin()+a);
                    cand.insert(cand.begin()+b, tmp);
                    mutated = true;
                }
            }
        } else if (moveType < 90){
            // ---- drop a project that scored 0 ----
            // Only meaningful if we have info; just drop a random one.
            int idx = rng() % cand.size();
            // Prefer dropping zero-score projects.
            for (int t = 0; t < 5; ++t){
                int k = rng() % cand.size();
                if (k < (int)sr.projScore.size() && sr.projScore[k] == 0){
                    idx = k; break;
                }
            }
            cand.erase(cand.begin()+idx);
            mutated = true;
        } else {
            // ---- try to insert a currently-not-assigned project ----
            // Pick a random unassigned project with a chance proportional to score.
            int tries = 30;
            while (tries-- > 0){
                int pid = rng() % P_;
                if (inSol[pid]) continue;
                // Try to build an assignment greedily from currently-available contributors.
                // Use the *original* skill levels as a conservative lower bound; the
                // simulator will verify validity at the actual time this project is
                // scheduled.
                const auto& req = PRJ[pid].req;
                vector<int> cs(req.size(), -1);
                vector<char> taken(C_, 0);
                // For projects that are already in cand, the contributors
                // assigned there are fine to reuse here (they'll just do it
                // later in time); we only need to avoid duplicates inside THIS
                // project.
                // Greedy, roles sorted by required level descending.
                vector<int> order(req.size());
                iota(order.begin(), order.end(), 0);
                sort(order.begin(), order.end(),
                     [&](int a, int b){ return req[a].second > req[b].second; });
                bool ok = true;
                // Track best "mentor skill" found so far for each skill to allow
                // mentoring when placing a lower-skilled contributor.
                unordered_map<int,int> teamMax;
                for (int r : order){
                    int s = req[r].first, l = req[r].second;
                    int chosen = -1;
                    // Need a contributor with level >= l; or level == l-1 if
                    // someone *already chosen* has >= l for skill s.
                    // First: exact/above match.
                    const auto& pool = contribBySkill[s];
                    for (int c : pool){
                        if (taken[c]) continue;
                        if (CON[c].skill[s] >= l){ chosen = c; break; }
                    }
                    if (chosen < 0 && teamMax[s] >= l){
                        // allow mentorable contributor (level l-1, including 0 if l==1)
                        if (l == 1){
                            for (int c = 0; c < C_; ++c){
                                if (!taken[c] && CON[c].skill[s] == 0){
                                    chosen = c; break;
                                }
                            }
                        } else {
                            for (int c : pool){
                                if (taken[c]) continue;
                                if (CON[c].skill[s] == l-1){ chosen = c; break; }
                            }
                        }
                    }
                    if (chosen < 0){ ok = false; break; }
                    cs[r] = chosen;
                    taken[chosen] = 1;
                    if (CON[chosen].skill[s] > teamMax[s])
                        teamMax[s] = CON[chosen].skill[s];
                }
                if (!ok) continue;
                // Insert at the end (simplest); neighbor moves may push it earlier.
                cand.emplace_back(pid, std::move(cs));
                inSol[pid] = 1;
                mutated = true;
                break;
            }
        }

        if (!mutated) continue;

        SimResult nr = simulate(cand);
        if (!nr.valid) continue;

        long long delta = nr.score - sr.score;
        bool accept = false;
        if (delta >= 0) accept = true;
        else {
            double pr = exp((double)delta / temperature);
            if (pr > (double)(rng() & 0xffff) / 65535.0) accept = true;
        }

        if (accept){
            // Fix inSol tracking for "drop" and "insert" moves (swap/replace
            // moves keep the project set unchanged).
            if (cand.size() != cur.size()){
                fill(inSol.begin(), inSol.end(), 0);
                for (auto& a : cand) inSol[a.first] = 1;
            }
            cur = std::move(cand);
            sr  = nr;
            if (sr.score > bestScore){
                bestScore = sr.score;
                best = cur;
                ++improves;
            }
        }

        // Cooling
        if ((iters & 1023) == 0){
            temperature *= 0.999;
            if (temperature < 0.01) temperature = 0.01;
        }
        if (now_sec() >= nextReport){
            cerr << "t=" << (int)(now_sec()-startT)
                 << "s  iters=" << iters
                 << "  cur=" << sr.score
                 << "  best=" << bestScore
                 << "  T=" << temperature
                 << "  |sol|=" << cur.size() << "\n";
            nextReport += 10.0;
        }
    }

    cerr << "DONE. iters=" << iters << "  improves=" << improves
         << "  best=" << bestScore
         << "  initial=" << (long long)0 /*placeholder*/ << "\n";

    // Write best.
    ofstream out(argv[3]);
    out << best.size() << "\n";
    for (auto& a : best){
        out << PRJ[a.first].name << "\n";
        for (size_t i = 0; i < a.second.size(); ++i){
            if (i) out << ' ';
            out << CON[a.second[i]].name;
        }
        out << "\n";
    }
    return 0;
}
