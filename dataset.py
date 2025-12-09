from collections import defaultdict
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import pandas as pd


class ImageTextContrastiveDataset(Dataset):

    def __init__(self, csv_path, image_size=128):
        super().__init__()
        self.data = pd.read_csv(csv_path)
        self.image_size = image_size

    def _safe_squeeze(self, img):
        """Squeeze without removing channel dim for 3D/4D images"""
        while img.ndim > 4:
            img = img.squeeze(0)
        return img

    def pad_img(self, img, size=128):
        try:
            img = self._safe_squeeze(img)
            if img.ndim == 4:
                img = img.mean(dim=0)
            
            if img.ndim != 3:
                raise ValueError(f"Image must be 3D after squeezing but got shape {img.shape}")
            orig_dtype = img.dtype
            
            img = img.float()
            img_min = img.min()
            img = (img - img_min) / (img.max() - img_min + 1e-8)
            
            x, y, z = img.shape
            max_size = max(x, y, z)
            scale_factor = size / max_size
            new_size = [int(d * scale_factor) for d in (x, y, z)]
            
            img = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            img = F.interpolate(img, size=new_size, mode='trilinear', align_corners=False)
            img = torch.clamp(img, -10, 10)
            # Pad to target size
            pad_x = (size - new_size[0]) // 2
            pad_y = (size - new_size[1]) // 2
            pad_z = (size - new_size[2]) // 2
            
            padded_img = torch.zeros((1, 1, size, size, size), dtype=img.dtype)
            padded_img[:, :, 
                      pad_x:pad_x + new_size[0],
                      pad_y:pad_y + new_size[1], 
                      pad_z:pad_z + new_size[2]] = img
            
            # Restore original value range
            padded_img = padded_img * (img.max() - img_min) + img_min
            return padded_img.to(orig_dtype)
            
        except Exception as e:
            print(f"Error in pad_img: {str(e)}")
            raise

    def norm_img(self, img):
        img = img.float()
        img = torch.nan_to_num(img, nan=0.0, posinf=img.max(), neginf=img.min())
        
        # Clip extrême + normalisation robuste
        img = torch.clamp(img, -3, 3)  # Limite dynamique
        median = torch.median(img)
        mad = torch.median(torch.abs(img - median))  # Écart médian absolu
        img = (img - median) / (mad + 1e-6)
        return img

    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        # Load MRI
        mri_array = sitk.GetArrayFromImage(sitk.ReadImage(row['mri_path']))
        mri_img = torch.from_numpy(mri_array).float() 
        mri_img = self.norm_img(mri_img)
        mri_img = self.pad_img(mri_img, self.image_size)
        
        # Load PET with 4D handling
        pet_array = sitk.GetArrayFromImage(sitk.ReadImage(row['pet_path']))
        pet_img = torch.from_numpy(pet_array).float()
        if pet_img.ndim == 4:
            # Take frame with maximum average intensity
            frame_means = pet_img.mean(dim=(1,2,3))
            pet_img = pet_img[frame_means.argmax()]
        pet_img = self.norm_img(pet_img)
        pet_img = self.pad_img(pet_img, self.image_size)
        if torch.isnan(mri_img).any() or torch.isinf(mri_img).any():
            raise ValueError(f"MRI image at index {index} contains NaN/inf")
        if torch.isnan(pet_img).any() or torch.isinf(pet_img).any():
            raise ValueError(f"PET image at index {index} contains NaN/inf")

        # Text input and report target
        clinical_text = row['clinical_text']
        diagnostic_report = row['diagnostic_report']

        return {
            'mri': mri_img,
            'pet': pet_img,
            'clinical_text':  row['clinical_text'],
            'reports': row['diagnostic_report']
        }

    def __len__(self):
        return len(self.data)


class ImageTextContrastiveCollator:
    def __init__(self):
        return
    def __call__(self, batch):
        inputs = defaultdict(list)
        for sample in batch:
            inputs['mri'].append(sample['mri'])
            inputs['pet'].append(sample['pet'])
            inputs['clinical_text'].append(sample['clinical_text']) # Changed key
            inputs['reports'].append(sample['reports'])  
        
        inputs['mri'] = torch.cat(inputs['mri'], dim=0)
        inputs['pet'] = torch.cat(inputs['pet'], dim=0)

        return inputs
    


class ZeroShotImageDataset(Dataset):

    def __init__(self, csv_path, image_size=128):
        super().__init__()
        self.data = pd.read_csv(csv_path)
        self.image_size = image_size

    def _safe_squeeze(self, img):
        """Squeeze without removing channel dim for 3D/4D images"""
        while img.ndim > 4:
            img = img.squeeze(0)
        return img

    def pad_img(self, img, size=128):
        try:
            img = self._safe_squeeze(img)
            
            # Handle 4D PET (t,z,y,x) by taking mean across time
            if img.ndim == 4:
                img = img.mean(dim=0)
            
            if img.ndim != 3:
                raise ValueError(f"Image must be 3D after squeezing but got shape {img.shape}")

            # Store original dtype
            orig_dtype = img.dtype
            
            # Convert to float32 for processing
            img = img.float()
            
            # Normalize to [0,1] temporarily for interpolation
            img_min = img.min()
            img = (img - img_min) / (img.max() - img_min + 1e-8)
            
            # Resize maintaining aspect ratio
            x, y, z = img.shape
            max_size = max(x, y, z)
            scale_factor = size / max_size
            new_size = [int(d * scale_factor) for d in (x, y, z)]
            
            img = img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            img = F.interpolate(img, size=new_size, mode='trilinear', align_corners=False)
            img = torch.clamp(img, -10, 10)
            # Pad to target size
            pad_x = (size - new_size[0]) // 2
            pad_y = (size - new_size[1]) // 2
            pad_z = (size - new_size[2]) // 2
            
            padded_img = torch.zeros((1, 1, size, size, size), dtype=img.dtype)
            padded_img[:, :, 
                      pad_x:pad_x + new_size[0],
                      pad_y:pad_y + new_size[1], 
                      pad_z:pad_z + new_size[2]] = img
            
            # Restore original value range
            padded_img = padded_img * (img.max() - img_min) + img_min
            return padded_img.to(orig_dtype)
            
        except Exception as e:
            print(f"Error in pad_img: {str(e)}")
            raise

    def norm_img(self, img):
        img = img.float()
        img = torch.nan_to_num(img, nan=0.0, posinf=img.max(), neginf=img.min())
        
        # Clip extrême + normalisation robuste
        img = torch.clamp(img, -3, 3)  # Limite dynamique
        median = torch.median(img)
        mad = torch.median(torch.abs(img - median))  # Écart médian absolu
        img = (img - median) / (mad + 1e-6)
        return img


    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        # Load MRI
        mri_array = sitk.GetArrayFromImage(sitk.ReadImage(row['mri_path']))
        mri_img = torch.from_numpy(mri_array).float() 
        mri_img = self.norm_img(mri_img)
        mri_img = self.pad_img(mri_img, self.image_size)
        
        # Load PET with 4D handling
        pet_array = sitk.GetArrayFromImage(sitk.ReadImage(row['pet_path']))
        pet_img = torch.from_numpy(pet_array).float()
        if pet_img.ndim == 4:
            # Take frame with maximum average intensity
            frame_means = pet_img.mean(dim=(1,2,3))
            pet_img = pet_img[frame_means.argmax()]
        pet_img = self.norm_img(pet_img)
        pet_img = self.pad_img(pet_img, self.image_size)
        if torch.isnan(mri_img).any() or torch.isinf(mri_img).any():
            raise ValueError(f"MRI image at index {index} contains NaN/inf")
        if torch.isnan(pet_img).any() or torch.isinf(pet_img).any():
            raise ValueError(f"PET image at index {index} contains NaN/inf")

        # Text input and report target
        clinical_text = row['clinical_text']
        diagnostic_report = row['diagnostic_report']

        return {
            'mri': mri_img,
            'pet': pet_img,
            'clinical_text': row['clinical_text'],
            'reports': row['diagnostic_report']
        }

    def __len__(self):
        return len(self.data)

class ZeroShotImageCollator:
    def __init__(self):
        return
    def __call__(self, batch):
        inputs = defaultdict(list)
        for sample in batch:
            inputs['mri'].append(sample['mri'])
            inputs['pet'].append(sample['pet'])
            inputs['clinical_text'].append(sample['clinical_text']) 
            inputs['reports'].append(sample['reports'])  

        inputs['mri'] = torch.cat(inputs['mri'], dim=0)
        inputs['pet'] = torch.cat(inputs['pet'], dim=0)


        return inputs
