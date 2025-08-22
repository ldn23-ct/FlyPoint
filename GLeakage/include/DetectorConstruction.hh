#ifndef DETECTORCONSTRUCTION_HH
#define DETECTORCONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4SystemOfUnits.hh"
#include "G4Material.hh"
#include "G4ExtrudedSolid.hh"
#include "G4UnionSolid.hh"
#include "G4SubtractionSolid.hh"
#include "G4Tubs.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"
class G4VPhysicalVolume;
class G4LogicalVolume;
class G4Box;
class G4Material;

class DetectorConstruction: public G4VUserDetectorConstruction
{
public:
    DetectorConstruction();
    ~DetectorConstruction() override;

    // G4LogicalVolume* GetLogTrapezoid() { return trapezoid_lv; }
    // G4LogicalVolume* GetLogBox2() { return box2_lv; }
    // G4LogicalVolume* GetLogCopperRing() { return ring_lv; }
    // G4LogicalVolume* GetLogCombinedlv() { return combined_lv; }

    G4VPhysicalVolume* Construct() override;
    void ConstructSDandField() override;
private:

    void VisAttributes();
    //DetectorConstruction* detector = nullptr;

    G4Material* fAir = nullptr;
    G4Material* fPb = nullptr;
    G4Material* fCu = nullptr;
    G4Material* fW = nullptr;
    G4Material* fGalactic = nullptr;

    // 几何体 Solid
    G4ExtrudedSolid* trapezoid_sv = nullptr;
    G4Box* box1_sv = nullptr;
    G4Box* hole_sv = nullptr; // 如果需要第二个盒子，可以定义
    G4UnionSolid* combined_sv = nullptr;
    G4SubtractionSolid* final_sv = nullptr;
    G4Box* box2_sv = nullptr;
    G4Tubs* ring_sv = nullptr;
    G4Tubs* cyl1_solid = nullptr;
    G4Box* hole_solid = nullptr;
    G4Box* cut_box = nullptr;
    G4SubtractionSolid* cyl1_with_cut = nullptr;
    G4SubtractionSolid* cyl1_with_hole = nullptr;
    G4Tubs* cyl2_solid = nullptr;
    G4UnionSolid* combined_solid = nullptr;

    // 几何体 Logical
    G4LogicalVolume* world_lv = nullptr;
    G4LogicalVolume* trapezoid_lv = nullptr;
    G4LogicalVolume* airbox0_lv = nullptr;
    G4LogicalVolume* box1_lv = nullptr;
    G4LogicalVolume* airbox1_lv = nullptr;
    G4LogicalVolume* box2_lv = nullptr;
    G4LogicalVolume* airbox2_lv = nullptr;
    G4LogicalVolume* cylCut_lv = nullptr;
    G4LogicalVolume* trd_lv = nullptr;
    G4LogicalVolume* ring_lv = nullptr;
    G4LogicalVolume* sd_lv = nullptr;

    // 几何尺寸（单位 mm）
    G4double trapezoid_bottom    = 68.4 * mm; // 下底长度
    G4double trapezoid_top       = 50.0 * mm; // 上底长度
    G4double trapezoid_height    = 10.0 * mm; // 梯形高度（Z 方向）
    G4double trapezoid_thickness = 44.0 * mm; // 拉伸长度

    G4double box1_y = 36.0 * mm;
    G4double box1_x = 11.0 * mm;
    G4double box1_z = 90.0; // = 16mm，与梯形一致

    G4double box2_y = 76.6 * mm;
    G4double box2_x = 20 * mm;
    G4double box2_z = 6 * mm;

    // 孔的截面尺寸（X方向厚度，Z方向宽度），沿Y轴贯穿，长度足够大以确保完全穿透
    G4double hole_x = 5 * mm;  // X方向宽度
    G4double hole_y = 30 * mm; // Y方向长度，远大于被打孔体长度，保证贯穿
    G4double hole_z = 1000 * mm;  // Z方向宽度

    G4double box3_y = 37.0 * mm;
    G4double box3_x = 5 * mm;
    G4double box3_z = 60 * mm;

    G4double innerRadius = 140.0 * mm; //避免碰撞增加2mm
    G4double outerRadius = 160.0 * mm;
    G4double halfLengthZ = 10.0 * mm;  

    G4double r1 = 50*mm;
    G4double h1 = 70*mm; // full height

    G4double hole_half = 10*mm;

    G4double cut_cube_x = 100 * mm;
    G4double cut_cube_y = 14 * mm;
    G4double cut_cube_z = 70 * mm;

    G4double r2 = 62.25*mm;
    G4double h2 = 209*mm;

};

#endif