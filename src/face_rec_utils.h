#ifndef _FACE_REC_UTILS_H_
#define _FACE_REC_UTILS_H_



extern Mat norm_0_255(InputArray _src);

extern void sigchld_handler(int s);

extern void *get_in_addr(struct sockaddr *sa);

extern enum_err_server read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';');

extern std::string base64_decode(std::string const& encoded_string);

extern inline bool is_base64(unsigned char c) ;

extern std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len);



#endif /* _FACE_REC_UTILS_H_*/
