#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <unordered_set>
#include <cstring>
#include <stop_token>

#include <yy981/time.h>

namespace fs = std::filesystem;


int cmd(const std::string& str) {
	return std::system(str.c_str());
}

uint32_t crc32(const uint8_t* data, size_t len) {
	static uint32_t table[256];
	static bool init = false;

	if (!init) {
		for (uint32_t i = 0; i < 256; ++i) {
			uint32_t c = i;
			for (int j = 0; j < 8; ++j)
				c = (c & 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1);
			table[i] = c;
		}
		init = true;
	}

	uint32_t crc = 0xFFFFFFFF;
	for (size_t i = 0; i < len; ++i)
		crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);

	return crc ^ 0xFFFFFFFF;
}


void upscale(std::stop_token st) {
	uint64_t pos;
	std::vector<uint32_t> indexReq;
	uint32_t h_t;
	std::ifstream ifsIndexReq("progress/indexReq.bin", std::ios::binary);
	std::ifstream ifsProgress("progress/progress.bin", std::ios::binary);
	while (ifsIndexReq.read(reinterpret_cast<char*>(&h_t), sizeof(h_t))) indexReq.push_back(h_t);
	ifsProgress.read(reinterpret_cast<char*>(&pos), sizeof(pos));
	
	size_t indexReqSize = indexReq.size();
	#pragma omp parallel for
	for (; pos < indexReqSize; ++pos) {
		std::cout << std::to_string(pos+1) << "/" << std::to_string(indexReqSize) << "\n";
		int result = cmd("py -3.13 upscale.py " + std::to_string(indexReq[pos]) + ".jpg");
		if (result != 0) {
			std::cerr << "\nupscale.py - [ERROR]: " << std::to_string(pos+1) << "(index=" << std::to_string(pos) << "\n\n";
		}
		if (st.stop_requested()) {
			std::ofstream ofsProgress("progress/progress.bin", std::ios::binary);
			ofsProgress.write(reinterpret_cast<const char*>(&pos), sizeof(pos));
			return;
		}
	}
}

void upscale_man() {
	std::jthread upscale_t(upscale);
	while (true) {
		std::cin.get();
		std::cout << "本当に中断しますか? [y,n]: ";
		char c;
		std::cin >> c;
		switch (c) {
			case 'Y': case 'y': return;
			default: {
				std::cout << "操作を取り消しました";
				continue;
			}
		}
	}
}

void checkHash() {
	std::cout << "本当に初期化しますか? [y,n,s(frame生成+input削除無し)]: ";
	bool skipGenFrame = false;
	char c;
	std::cin >> c;
	switch (c) {
		case 'Y': case 'y': break;
		case 'S': case 's': skipGenFrame = true; break;
		default: {
			std::cout << "操作を取り消しました\n";
			exit(1);
		}
	}
	
	const std::vector<std::string> refreshDirs {
		"input", "output", "outputFrames", "progress"
	};
	if (!skipGenFrame) for (const std::string d: refreshDirs) {
		fs::remove_all(d);
		fs::create_directory(d);
	}
	
	
	if (!skipGenFrame) if (cmd("call frame.bat")) throw std::runtime_error("frame.bat");
	
	constexpr size_t BUF = 1024 * 1024;
	std::vector<uint8_t> buffer(BUF);
	std::unordered_set<uint32_t> indexReq;
	std::vector<uint32_t> index;

	for (auto& e : fs::directory_iterator("input")) {
		std::ifstream ifs(e.path(), std::ios::binary);
		std::fill(buffer.begin(), buffer.end(), 0);

		ifs.read((char*)buffer.data(), buffer.size());
		uint32_t h = crc32(buffer.data(), ifs.gcount());
		ifs.close();
		index.push_back(h);
		if (indexReq.count(h)) {
			// 既に存在するCRC → 重複扱い
			fs::remove(e.path());
		} else {
			// 新しいCRC → ファイル名を書き換え
			auto newPath = e.path().parent_path() / (std::to_string(h) + e.path().extension().string());
			for (int attempt = 0; attempt < 3; ++attempt) {
				try {
					fs::rename(e.path(), newPath);
					break; // 成功したら抜ける
				} catch (const fs::filesystem_error& err) {
					if (attempt == 2) throw; // 3回失敗したら例外を投げる
					sleepc(tu::l,100);
				}
			}
			indexReq.insert(h);
		}
	}
	
	const uint64_t pos = 0;
	std::ofstream ofsIndex("progress/index.bin", std::ios::binary);
	for (uint32_t h: index) ofsIndex.write(reinterpret_cast<const char*>(&h), sizeof(h));
	std::ofstream ofsIndexReq("progress/indexReq.bin", std::ios::binary);
	for (uint32_t h: indexReq) ofsIndexReq.write(reinterpret_cast<const char*>(&h), sizeof(h));
	std::ofstream ofsProgress("progress/progress.bin", std::ios::binary);
	ofsProgress.write(reinterpret_cast<const char*>(&pos), sizeof(pos));
}

int main(int argc, char* argv[]) {
	if (argc != 2) throw std::runtime_error("argc != 2");
	if (!strcmp(argv[1],"init")) checkHash();
	else if (!strcmp(argv[1],"proc")) upscale_man();
	else throw std::runtime_error("argv[1] == ?");
}

/*

init -> index.bin -> upscale
upscale -> progress.bin -> upscale
index.bin = vector(重複在り)
indexReq.bin = init->unordered_set(重複無し) upscale->vector(重複無し)
progress = uint64_t

*/