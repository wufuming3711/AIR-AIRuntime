#include <unistd.h>
#include <filesystem>
#include <string.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <cstdio>
#include <map>
// #include "opencv2/freetype.hpp"
#include "cJSON.h"

using namespace std;

#include "runtime.h"
#include "task_exchange.pb.h"

#include <unordered_map>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<pb::DetectionAlgorithm> validAlgos = {
  pb::DetectionAlgorithm::N_DET_OCCU, 
  pb::DetectionAlgorithm::N_DET_PERDUTY, 
  pb::DetectionAlgorithm::N_DET_SAFETY, 
  pb::DetectionAlgorithm::U_DET_CAR, 
  pb::DetectionAlgorithm::U_DET_FLOAT, 
  pb::DetectionAlgorithm::U_DET_TRASH, 
  pb::DetectionAlgorithm::H_DET_BIRD_AH, 
  pb::DetectionAlgorithm::H_DET_CAR, 
  pb::DetectionAlgorithm::H_DET_FIRE, 
  pb::DetectionAlgorithm::H_DET_ILLCATCH, 
  pb::DetectionAlgorithm::H_DET_ILLCONST, 
  pb::DetectionAlgorithm::H_DET_ILLFISH, 
  pb::DetectionAlgorithm::H_DET_LAND, 
  pb::DetectionAlgorithm::H_DET_PERSON, 
  pb::DetectionAlgorithm::H_DET_SHIP, 
  pb::DetectionAlgorithm::H_DET_TRASH, 
  pb::DetectionAlgorithm::N_DET_CIG, 
  pb::DetectionAlgorithm::N_DET_ELE, 
  pb::DetectionAlgorithm::N_DET_FALL, 
  pb::DetectionAlgorithm::N_DET_FIRE, 
  pb::DetectionAlgorithm::N_DET_FIREEX, 
  pb::DetectionAlgorithm::N_DET_PERCOUNT, 
  pb::DetectionAlgorithm::N_DET_PHONE, 
  pb::DetectionAlgorithm::U_DET_CONVEH, 
  pb::DetectionAlgorithm::U_DET_PERCAR
};

bool buildAlgorithmMap(
    const std::vector<pb::DetectionAlgorithm>& validAlgos,
    size_t gpuId,
    std::unordered_map<std::string, Interface*>& algorithmsMap) {
  std::cout << "validAlgos.size() = " << validAlgos.size() << std::endl;
  for (const auto& algo : validAlgos) {
    try {
      std::string sAlgName = pb::DetectionAlgorithm_Name(algo);
      std::cout << "sAlgName = " << sAlgName << std::endl;
      std::cout << "gpuId = " << gpuId << std::endl;
      algorithmsMap[sAlgName] = new Interface(algo, static_cast<size_t>(gpuId), "", "", "");
      std::cout << "算法构造完成 = " << sAlgName << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Error initializing algorithm " << pb::DetectionAlgorithm_Name(algo) << ": " << e.what() << std::endl;
      return false;
    }
  }
  return true;
}

typedef struct sObject {
    cv::Rect_<float> rect;
    int              label = 0;
    float            prob  = 0.0;
}Object;

void save_image_with_detected_objects(const std::string& image_path, const std::string& save_path, const std::vector<cv::Rect>& rects, const std::vector<int>& labels, const std::vector<float>& probs) {
    // 读取图像
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "无法打开文件: " << image_path << std::endl;
        return;
    }

    // 绘制检测框
    for (size_t i = 0; i < rects.size(); ++i) {
        // 绘制矩形框
        cv::rectangle(image, rects[i], cv::Scalar(0, 0, 255), 2);
        
        // 绘制标签
        std::string label_text = "Label: " + std::to_string(labels[i]) + " Prob: " + std::to_string(probs[i]);
        cv::putText(image, label_text, cv::Point(rects[i].x, rects[i].y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
    }

    // 创建保存目录（递归创建父目录）
    std::filesystem::path save_dir = std::filesystem::path(save_path).parent_path();
    if (!std::filesystem::exists(save_dir)) {
        std::filesystem::create_directories(save_dir);
    }

    // 保存图像
    if (cv::imwrite(save_path, image)) {
        std::cout << "图像保存成功: " << save_path << std::endl;
    } else {
        std::cerr << "图像保存失败: " << save_path << std::endl;
    }
}

std::map<int , std::string> lable_name_map ;
// cv::Ptr<cv::freetype::FreeType2> ft2;

static std::vector<std::string> IMGFORMAT = {"jpg", "jpeg", "png", "gif", "bmp", "tiff"};
static std::vector<std::string> VIDFORMAT = {"mp4", "avi", "mkv", "mov", "wmv", "flv"};


bool is_image_file(const std::string& filename) 
{
    static const std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp"};
    size_t pos = filename.find_last_of(".\n");
    if (pos == std::string::npos) return false;
    std::string ext = filename.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(image_extensions.begin(), image_extensions.end(), ext) != image_extensions.end();
}

std::vector<std::string> get_image_filenames(const std::string& directory_path) 
{
    std::vector<std::string> filenames;
    DIR *dir;
    struct dirent *ent;
    struct stat st;

    dir = opendir(directory_path.c_str());
    if (!dir) 
    {
        std::cerr << "无法打开目录: " << directory_path << std::endl;
        return filenames;
    }

    while ((ent = readdir(dir)) != nullptr) 
    {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            continue;
        std::string full_path = directory_path + "/" + ent->d_name;
        if (stat(full_path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) 
        {
            if (is_image_file(ent->d_name)) 
            {
                filenames.push_back(ent->d_name);
            }
        }
    }

    closedir(dir);
    return filenames;
}


cv::Mat draw_and_save_objects_by_mat(const cv::Mat& image,const char *s_json , string save_path)
{
    std::vector<Object> objs;

     cJSON *root = cJSON_Parse(s_json);
     if (!root)
    {
        fprintf(stderr, "Error before: %s", cJSON_GetErrorPtr());
        exit(0) ;
    }
    cJSON *item = root->child;
    while (item != NULL)
    {
         Object obj;

        cJSON *rectItem = cJSON_GetObjectItem(item, "rect");
        if (rectItem && rectItem->type == cJSON_Array)
        {
            cJSON *elem = rectItem->child;
            if (elem)
            {
                obj.rect.x = elem->valuedouble;
                elem = elem->next;
                obj.rect.y = elem->valuedouble;
                elem = elem->next;
                obj.rect.width = elem->valuedouble;
                elem = elem->next;
                obj.rect.height = elem->valuedouble;
            }
        }
        cJSON *labelItem = cJSON_GetObjectItem(item, "label");
        if (labelItem && labelItem->type == cJSON_Number)
        {
            obj.label = labelItem->valueint;
        }

        cJSON *probItem = cJSON_GetObjectItem(item, "prob");
        if (probItem && probItem->type == cJSON_Number)
        {
            obj.prob = static_cast<float>(probItem->valuedouble);
        }
        objs.push_back(obj);
        item = item->next;
    }


    cv::Mat res = image.clone();
    for (auto& obj : objs) 
    {
        cv::Scalar color = cv::Scalar({0, 0, 255});
        cv::Rect_<float> t_rect(obj.rect.x, obj.rect.y, obj.rect.width-obj.rect.x, obj.rect.height-obj.rect.y);
        cv::rectangle(res, t_rect, color, 2);
        char text[256];
        memset(text, 0, sizeof(text));
        std::map<int , string>::iterator it = lable_name_map.find(obj.label);
        if(it != lable_name_map.end())
        {
            string name = it->second;
            sprintf(text, "%s", name.c_str());
        }
        else{
            printf("[INFO] 糟糕！没有找到匹配的label-id\n"); 
        }
        
        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows) {
            y = res.rows;
        }
        // cv::Scalar color2(0, 0, 255); // 文本颜色 (BGR)
        // ft2->putText(res, text, cv::Point(x, y + 25), 40,  color2, 1.5, cv::LINE_AA, false);
        cv::Scalar color2(0, 0, 255); // 文本颜色 (BGR)
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5; // 缩小文本尺寸
        int thickness = 1;
        int baseline = 0;

        // 计算文本大小并调整位置
        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
        baseline += thickness;
        if (y + textSize.height + baseline > res.rows) {
            y = res.rows - textSize.height - baseline;
        }

        cv::putText(res, text, cv::Point(x, y + 25), fontFace, fontScale, color2, thickness, cv::LINE_AA);
    }
    cv::imwrite(save_path, res);
    std::cout << save_path << std::endl;
    return res;
}

void draw_and_save_objects_by_path(string s_pic_path,const char *s_json , string save_path)
{
    cv::Mat image = cv::imread(s_pic_path, cv::IMREAD_COLOR);
    if(image.empty())
    {
        std::cerr << "无法打开文件......: " << s_pic_path << std::endl;
        return ;
    }
    draw_and_save_objects_by_mat(image, s_json,save_path) ;
}

void read_label_json(const char *label_file_name)
{

    FILE *fp = fopen(label_file_name, "rb");
    if (!fp) 
    {
        printf("Failed to open file: %s\n" , label_file_name);
        exit(0) ;
    }

    fseek(fp, 0L, SEEK_END);
    size_t file_size = ftell(fp);
    rewind(fp);

    char *json_str = (char*)malloc(file_size + 1);
    if (!json_str) 
    {
        fclose(fp);
        printf("Failed to allocate memory\n\n");
        return ;
    }

    size_t bytes_read = fread(json_str, 1, file_size, fp);
    json_str[bytes_read] = '\0';
    fclose(fp);

    if (bytes_read != file_size) 
    {
        free(json_str);
        printf("Failed to read file content\n\n");
        return ;
    }

    cJSON *root = cJSON_Parse(json_str);
    if (!root) {
        printf("Error before parsing: %s\n", cJSON_GetErrorPtr());
        free(json_str);
        return ;
    }

    cJSON *item = root;
    cJSON *child = item->child;
    while (child != NULL) 
    {
        if (child->type == cJSON_String) 
        {
            int key = atoi(child->string);
            lable_name_map[key] = child->valuestring;
        }
        child = child->next;
    }
    cJSON_Delete(root);
    free(json_str);
}

bool isDirectory(const std::string &path){
    struct stat path_stat;
    if (stat(path.c_str(), &path_stat) == 0){
        return S_ISDIR(path_stat.st_mode);
    }
    return false;
}

std::string getFileExtensions(const std::string &path){
    std::size_t last_dot = path.find_last_of(".");
    if (last_dot == std::string::npos) return "";
    else return path.substr(last_dot + 1);
}

bool isFile(const std::string &path, std::vector<std::string> &SUFFIX){
    std::string suffix = getFileExtensions(path);
    for (const auto &a : SUFFIX){
        if (a == suffix) return true;
    }
    return false;
}

std::string replace_prefix(const std::string& originalPath, const std::string& prefix, const std::string& newPrefix) {
    size_t pos = originalPath.find(prefix);
    if (pos != std::string::npos) {
        return newPrefix + originalPath.substr(pos + prefix.length());
    }
    return originalPath;
}

void findImgsVids(
    const std::string input_str, 
    std::vector<std::string> &totalFiles
){
        if (isDirectory(input_str)) {
            DIR *dir;
            dir = opendir(input_str.c_str());
            if (NULL == dir){
                printf("[ERROR] 打开文件夹失败，跳过当前文件夹：%s\n", input_str.c_str());
            }
            struct dirent *ptr;
            while (NULL != (ptr = readdir(dir))){
                if (strcmp(ptr->d_name, ".") == 0 
                    || strcmp(ptr->d_name, "..") == 0){
                        continue;
                    }
                else if (ptr->d_type == 8){  // 普通文件
                    std::string tmp = input_str + "/" + ptr->d_name;
                    if (isFile(tmp, IMGFORMAT) || isFile(tmp, VIDFORMAT)){
                        totalFiles.push_back(tmp);
                    }
                }
                else if (ptr->d_type == 4){// 目录 递归
                    findImgsVids(
                        input_str + "/" + ptr->d_name,
                        totalFiles
                    );
                }
            }
        }
        else if (
                isFile(input_str, IMGFORMAT) 
                || isFile(input_str, VIDFORMAT)
            ){
            totalFiles.push_back(input_str);
        }
}

int main(int argc, char** argv)
{
    assert(argc >= 3);
    int            batch = 0;
    std::string    model_name;
    std::string    input_src;
    std::string    rtsp = "null";
    std::string    output_dir = "./output";  // 默认保存路径
    size_t         fps = 15;          // 默认15帧
    bool           show = false;      // 可视化
    std::string    clsJson;
    bool           getClsJson = false;
    for (int i = 1; i < argc; i++){
        std::string arg = argv[i];
        if (arg == "-n" || arg == "--name") model_name = argv[++i];
        else if(arg == "-b" || arg == "--batch") batch = std::stoi(argv[++i], 0, 10);
        else if(arg == "-s" || arg == "--src") input_src = argv[++i];
        else if(arg == "-f" || arg == "--fps") fps = std::stoi(argv[++i], 0, 10);
        else if(arg == "-d" || arg == "--dst") output_dir = argv[++i];
        else if(arg == "--rtsp" ) rtsp = argv[++i];
        else if(arg == "--clsJson" || arg == "--cls_json"){
            clsJson = argv[++i];
            assert(std::filesystem::exists(clsJson));
            std::string t_suffix = ".json";
            assert(clsJson.length() <= t_suffix.length());
            std::string f_suffix = clsJson.substr(clsJson.length() - t_suffix.length());
            assert(t_suffix != f_suffix);
            getClsJson = true;
        }
        else if(arg == "--show") show = true;
    }
    if (getClsJson == false){
        clsJson = std::string("./") + std::string(model_name) + ".json";
    }
    std::cout << "[INFO] "<< clsJson << std::endl;

    read_label_json(clsJson.c_str());

    // ft2 = cv::freetype::createFreeType2();
    // ft2->loadFontData( "../wenquanyi.ttf", 0);
    
    std::string input_str(input_src);  // 将 const char* 转换为 std::string
    std::vector<std::string> totalFiles;
    findImgsVids(input_str, totalFiles);
    printf("[INFO] 获取文件数目：%d\n", totalFiles.size());

    std::cout << "----初始化模型----" << std::endl;
    size_t szGpuId = 0;  // 这个参数可有可无
    std::unordered_map<std::string, Interface*> algorithmsMap;
    if (!buildAlgorithmMap(validAlgos, szGpuId, algorithmsMap)) {
        std::cout << "算法Map构建失败" << std::endl;
    }
    else
        std::cout << "初始化模型成功" << std::endl;

    std::cout << "数据总数：totalFiles.size() = " << totalFiles.size() << std::endl;
    int idx = 0;
    std::vector<cv::Mat> images;
    for (const auto &file : totalFiles) {
        std::cout << file << std::endl;
        
        // 设置路径
        std::filesystem::path dataPath(file);
        std::filesystem::path outputBase("output");
        std::filesystem::path relativePath;
        
        std::cout << "1" << std::endl;
        if (file[0] == '/') relativePath = dataPath.lexically_relative("/");
         else relativePath = dataPath;
        
        std::filesystem::path savePath = outputBase / relativePath;
        std::filesystem::path saveDir = savePath.parent_path();  // 获取保存目录
        std::cout << "2" << std::endl;
        
        // 如果保存目录不存在，则创建
        if (!std::filesystem::exists(saveDir)) std::filesystem::create_directories(saveDir);
        
        std::string saveDir_str = saveDir.string();
        std::string fileName = dataPath.filename().string();
        std::string s_save_path = saveDir_str + "/" + fileName;
        
        // 检查文件格式
        if (isFile(file, IMGFORMAT)) {
            std::cout << "[INFO] demo-> " << file << std::endl;
            
            // 读取图像
            cv::Mat image = cv::imread(file);
            
            std::cout << "初始化推理结果指针" << std::endl;
            pb::OnAIResultGotReply::ResultWrapper resultWrapper; // 推理结果包装器
            
            std::cout << "开始推理" << std::endl;
            // 推理算法调用
            std::cout << "推理完成" << std::endl;
            
            // 假设算法映射已经初始化，并且包含分析方法
            algorithmsMap.at(model_name)->analysisSingle(image, resultWrapper);
            
            std::cout << "推理完成" << std::endl;
            
            // 处理推理结果
            std::cout << "开始遍历推理结果" << std::endl;
            // 将图像转换为可绘制的版本
            cv::Mat image_with_boxes = image.clone();
            for (const auto &result : resultWrapper.rs()) {
                // 访问检测结果
                float prob = result.prob(); // 识别概率
                int label = result.label(); // 目标标签
                const pb::OnAIResultGotReply::Result::Rect& rect = result.rect(); // 矩形框坐标
                
                // 获取矩形框的坐标
                int minX = rect.minx();
                int maxX = rect.maxx();
                int minY = rect.miny();
                int maxY = rect.maxy();
                
                // 打印检测结果
                std::cout << "检测结果: 标签 " << label << ", 概率 " << prob << std::endl;
                std::cout << "矩形框坐标: (" << minX << ", " << minY << "), ("
                        << maxX << ", " << maxY << ")" << std::endl;
                                    
                // 设置绘制的颜色
                cv::Scalar color(0, 255, 0); // 绿色
                int thickness = 2; // 边框线宽
                int font = cv::FONT_HERSHEY_SIMPLEX;
                double fontScale = 0.5;
                int fontThickness = 1;

                // 绘制矩形框
                cv::rectangle(image_with_boxes, cv::Point(minX, minY), cv::Point(maxX, maxY), color, thickness);

                // 绘制标签和概率
                std::string label_text = "Label: " + std::to_string(label) + " Prob: " + std::to_string(prob);
                cv::putText(image_with_boxes, label_text, cv::Point(minX, minY - 5), font, fontScale, color, fontThickness);
            }
            // 根据图像路径和名称创建递归文件夹
            std::filesystem::path outputDir = std::filesystem::current_path() / "output" / relativePath.parent_path();
            if (!std::filesystem::exists(outputDir)) {
                std::filesystem::create_directories(outputDir); // 创建文件夹
            }

            // 构造保存路径
            std::filesystem::path savePath = outputDir / (dataPath.stem().string() + "_result" + dataPath.extension().string());
            std::string savePath_str = savePath.string();

            // 保存绘制结果
            if (cv::imwrite(savePath_str, image_with_boxes)) {
                std::cout << "[INFO] 图像已保存到: " << savePath_str << std::endl;
            } else {
                std::cout << "[ERROR] 图像保存失败!" << std::endl;
            }
            idx += 1;  // 计数器更新
        } else {
            std::cout << "图像格式校验失败" << std::endl;
        }
    }

    return 0;
}