
#include "opencv2/opencv.hpp"
#include "tracy.hpp"

struct nx_bg_sub
{
    cv::Ptr<cv::BackgroundSubtractor> m_background_sub;
    bool m_is_ready;
};

nx_bg_sub nx_bg_sub_create(int32_t h,double vart , bool detectShadows) 
{
    nx_bg_sub ret = {};
    ret.m_background_sub = cv::createBackgroundSubtractorMOG2(h,vart,detectShadows);
    ret.m_is_ready = true;
    return ret ;
}

struct nx_mat
{
    cv::Mat m_mat;
};

bool nx_mat_is_empty(const nx_mat &mat)
{
    return mat.m_mat.empty();
}

struct nx_vmd_ocv_context 
{
    nx_mat m_prev_frame;
    nx_mat m_curr_frame;
    nx_mat m_diff_frame;
    nx_mat m_thresh_frame;
    nx_bg_sub m_bg_sub;
};
struct nx_vmd_ocv_params
{
    uint16_t m_bg_sub_history;
    uint16_t m_bg_sub_var_threshold;
    bool m_bg_sub_detect_shadows;
    uint8_t m_gaussian_blur_k_size;
    uint8_t m_threshold_min;
    uint8_t m_morph_ellipse_size;
    uint16_t m_obj_min_area;
    uint16_t m_obj_max_area;

};

void nx_mat_guassian_blur(nx_mat frame,uint8_t size)
{
    cv::GaussianBlur(frame.m_mat,frame.m_mat,cv::Size(size,size),0);
}
void nx_mat_thresh(nx_mat &res,nx_mat &img,uint32_t tfrom,uint32_t tto)
{
    cv::threshold(img.m_mat , res.m_mat ,tfrom,tto,cv::THRESH_BINARY);
}

void nx_vmd_ocv_apply_bg_sub(nx_vmd_ocv_context * context , nx_mat &frame,nx_mat & fgmask)
{
    if(context->m_bg_sub.m_is_ready)
    {
        context->m_bg_sub.m_background_sub->apply(frame.m_mat,fgmask.m_mat);
    }
}

void nx_mat_abs_diff(nx_mat & res , const nx_mat & a , const nx_mat & b)
{
    cv::absdiff(a.m_mat,b.m_mat,res.m_mat);
}
void nx_mat_bitwise_and(nx_mat &res , nx_mat &a , nx_mat &b)
{
    cv::bitwise_and(a.m_mat,b.m_mat,res.m_mat);
}

nx_mat nx_mat_get_morph_ellipse(uint32_t sx,uint32_t sy)
{
    nx_mat ret = {};
    ret.m_mat = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(sx,sy));
    return ret;
}

void nx_mat_morph_open(nx_mat &a, nx_mat &kern)
{
    cv::morphologyEx(a.m_mat,a.m_mat,cv::MORPH_OPEN,kern.m_mat);
}

void nx_mat_morph_close(nx_mat &a, nx_mat &kern)
{
    cv::morphologyEx(a.m_mat,a.m_mat,cv::MORPH_OPEN,kern.m_mat);
}
struct nx_vmd_ocv_bbox 
{
    cv::Rect box;
};


std::vector<nx_vmd_ocv_bbox> nx_vmd_ocv_get_object_bboxes(nx_mat mask,uint16_t min_obj_size, uint16_t max_obj_size)
{

    std::vector<nx_vmd_ocv_bbox> ret;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.m_mat,contours,cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for(auto &c : contours)
    {
        uint16_t l_area = cv::contourArea(c);

        if(l_area > min_obj_size && l_area < max_obj_size)
        {
            nx_vmd_ocv_bbox rbbox;
            rbbox.box = cv::boundingRect(c);
            ret.push_back(rbbox);
        }
    }
    return ret;
}
void nx_mat_copy(nx_mat &dest, const nx_mat &src)
{
    src.m_mat.copyTo(dest.m_mat);
}

void nx_mat_to_gray(nx_mat & a )
{
    cv::cvtColor(a.m_mat, a.m_mat, cv::COLOR_BGR2GRAY);
}

std::vector<nx_vmd_ocv_bbox> nx_vmd_ocv_detect_motion_2(nx_vmd_ocv_context* vmd_context, nx_vmd_ocv_params & params, nx_mat frame)
{
    ZoneScoped;
    {
		ZoneScopedN("cvt bgr 2 gray");
		nx_mat_to_gray(frame);
	}
    nx_mat fg_mask,result;
    {
        ZoneScopedN("blur and abs diff ");
        nx_mat blur;
        nx_mat_copy(blur,frame);
        nx_mat_guassian_blur(blur,params.m_gaussian_blur_k_size);
        nx_mat_abs_diff(fg_mask,blur,frame); 
    }

    vmd_context->m_curr_frame = frame;

    if(!nx_mat_is_empty(vmd_context->m_prev_frame))
    {
        {
            ZoneScopedN("abs diff");
            nx_mat_abs_diff(vmd_context->m_diff_frame,vmd_context->m_prev_frame,vmd_context->m_curr_frame);
        }
        {
            ZoneScopedN("thresh");
            nx_mat_thresh(vmd_context->m_thresh_frame,vmd_context->m_diff_frame, params.m_threshold_min ,255);
            //cv::imshow("thresh",vmd_context->m_thresh_frame.m_mat);
        }
        {
            ZoneScopedN("bitwise and");
            nx_mat_bitwise_and(result,fg_mask,vmd_context->m_thresh_frame);
        }

        {
            ZoneScopedN("morphology");
            nx_mat ker = nx_mat_get_morph_ellipse(params.m_morph_ellipse_size,params.m_morph_ellipse_size);
            nx_mat_morph_open(result,ker);
            //nx_mat_morph_close(result,ker);
            //cv::imshow("res",result.m_mat);
        }
        {
            ZoneScopedN("get contours and find bboxes");
            std::vector<nx_vmd_ocv_bbox> bboxes = nx_vmd_ocv_get_object_bboxes(result,params.m_obj_min_area,params.m_obj_max_area);
            nx_mat_copy(vmd_context->m_prev_frame, frame);
            return bboxes;
        }
    }
	nx_mat_copy(vmd_context->m_prev_frame, frame);
    std::vector<nx_vmd_ocv_bbox> bboxes; 
    return bboxes;


}
std::vector<nx_vmd_ocv_bbox> nx_vmd_ocv_detect_motion(nx_vmd_ocv_context* vmd_context, nx_vmd_ocv_params & params, nx_mat frame)
{
    ZoneScoped;
    if (!vmd_context->m_bg_sub.m_is_ready)
    {
        vmd_context->m_bg_sub = nx_bg_sub_create(params.m_bg_sub_history, params.m_bg_sub_var_threshold, params.m_bg_sub_detect_shadows);
    }
    {
		ZoneScopedN("cvt bgr 2 gray");
		nx_mat_to_gray(frame);
	}
    nx_mat fg_mask,result;
    {
        ZoneScopedN("blur");
        nx_mat_guassian_blur(frame,params.m_gaussian_blur_k_size);
    }
    {
        ZoneScopedN("apply bg sub");
        nx_vmd_ocv_apply_bg_sub(vmd_context,frame,fg_mask);
        vmd_context->m_curr_frame = frame;
    }
    if(!nx_mat_is_empty(vmd_context->m_prev_frame))
    {
        {
            ZoneScopedN("abs diff");
            nx_mat_abs_diff(vmd_context->m_diff_frame,vmd_context->m_prev_frame,vmd_context->m_curr_frame);
        }
        {
            ZoneScopedN("thresh");
            nx_mat_thresh(vmd_context->m_thresh_frame,vmd_context->m_diff_frame, params.m_threshold_min ,255);
        }
        {
            
            ZoneScopedN("bitwise and");
            nx_mat_bitwise_and(result,fg_mask,vmd_context->m_thresh_frame);
        }

        {
            ZoneScopedN("morphology");
            nx_mat ker = nx_mat_get_morph_ellipse(params.m_morph_ellipse_size,params.m_morph_ellipse_size);
            nx_mat_morph_open(result,ker);
            nx_mat_morph_close(result,ker);
        }
        {
            ZoneScopedN("get contours and find bboxes");
            std::vector<nx_vmd_ocv_bbox> bboxes = nx_vmd_ocv_get_object_bboxes(result,params.m_obj_min_area,params.m_obj_max_area);
            nx_mat_copy(vmd_context->m_prev_frame, frame);
            return bboxes;
        }
    }
	nx_mat_copy(vmd_context->m_prev_frame, frame);
    std::vector<nx_vmd_ocv_bbox> bboxes; 
    return bboxes;
}

int main(int argc , char ** argv)
{
    cv::VideoCapture cap(1);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    //cap.set(cv::CAP_PROP_CONVERT_RGB, false);
    //cap.set(cv::CAP_PROP_FORMAT, CV_8UC1);
    if(!cap.isOpened())
        return -1;
    nx_vmd_ocv_context vmdcontext = {};
    nx_vmd_ocv_params vmdparams =  {.m_bg_sub_history = 500,
                                    .m_bg_sub_var_threshold = 32 , 
                                    .m_bg_sub_detect_shadows = false,
                                    .m_gaussian_blur_k_size = 9 ,
                                    .m_threshold_min = 20,
                                    .m_morph_ellipse_size=2,
                                    .m_obj_min_area=1,
                                    .m_obj_max_area=500,
    };

    cv::Mat frame;

    while(cap.read(frame))
    {
        nx_mat nframe ={frame};
#if 0 
        std::vector<nx_vmd_ocv_bbox> bboxes = nx_vmd_ocv_detect_motion(&vmdcontext, vmdparams,nframe);
#else 
        std::vector<nx_vmd_ocv_bbox> bboxes = nx_vmd_ocv_detect_motion_2(&vmdcontext, vmdparams,nframe);
#endif
        for(auto & bbox : bboxes)
        {
            cv::rectangle(frame,bbox.box,cv::Scalar(0,255.0),2);
        }

        cv::imshow("MovingObjects",frame);
        if(cv::waitKey(30) == 27) break;

        FrameMark;

    }



    return 0;
}

