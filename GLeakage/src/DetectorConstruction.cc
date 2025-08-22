#include "DetectorConstruction.hh"
#include "Config.hh"
#include "AirSD.hh"
#include "globals.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4SystemOfUnits.hh"
#include "G4ExtrudedSolid.hh"
#include "G4UnionSolid.hh"
#include "G4SubtractionSolid.hh"
#include "G4Tubs.hh"
#include "G4OpticalSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4SDManager.hh"
#include "G4Sphere.hh"
#include "G4Trd.hh"

DetectorConstruction::DetectorConstruction()
{}

DetectorConstruction::~DetectorConstruction()
{}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    G4NistManager* nist = G4NistManager::Instance();
    fAir = nist->FindOrBuildMaterial("G4_AIR");
    fPb = nist->FindOrBuildMaterial("G4_Pb");
    fCu = nist->FindOrBuildMaterial("G4_Cu");
    fW = nist->FindOrBuildMaterial("G4_W");
    fGalactic = nist->FindOrBuildMaterial("G4_Galactic");

    // World
    G4double world_size = Config::Instance().GetWorldBoxL() * mm;
    auto world_sv = new G4Box("World", world_size/2, world_size/2, world_size/2);
    world_lv = new G4LogicalVolume(world_sv, fAir, "World");
    auto world_pv = new G4PVPlacement(nullptr, G4ThreeVector(), world_lv, "World", nullptr, false, 0, true);

    //SD
    G4double r1 = Config::Instance().GetR1();
    G4double r2 = Config::Instance().GetR2();
    G4Sphere* sd_sv = new G4Sphere("LeakageSphere", r1*mm, r2*mm, 0, 360*deg, 0, 180*deg);
    sd_lv = new G4LogicalVolume(sd_sv, fAir, "LeakageSphere");
    new G4PVPlacement(0, G4ThreeVector(), sd_lv, "LeakageSphere", world_lv, false, 0);  

    // // Test
    // auto t_sv = new G4Box("Test", 25*mm, 25*mm, 25*mm);
    // auto t_lv = new G4LogicalVolume(t_sv, fPb, "Test");
    // new G4PVPlacement(0, G4ThreeVector(), t_lv, "Test", sd_lv, false, 0);

    // 构建前准直系统
    // 定义梯形台
    std::vector<G4TwoVector> yzVertices;
    yzVertices.push_back(G4TwoVector(-0.5 * trapezoid_bottom, -0.5 * trapezoid_height));
    yzVertices.push_back(G4TwoVector( 0.5 * trapezoid_bottom, -0.5 * trapezoid_height));
    yzVertices.push_back(G4TwoVector( 0.5 * trapezoid_top,     0.5 * trapezoid_height));
    yzVertices.push_back(G4TwoVector(-0.5 * trapezoid_top,     0.5 * trapezoid_height));

    auto trapezoid_sv = new G4ExtrudedSolid("Trapezoid",
                                       yzVertices,
                                       0.5 * trapezoid_thickness,
                                       G4TwoVector(0,0), 1.0,
                                       G4TwoVector(0,0), 1.0);
    trapezoid_lv = new G4LogicalVolume(trapezoid_sv, fPb, "Trapezoid");
    G4RotationMatrix rot;
    rot.rotateY(90*deg);
    rot.rotateX(90*deg);
    new G4PVPlacement(G4Transform3D(rot, G4ThreeVector(0, 0, 36.0*mm + trapezoid_height/2)), trapezoid_lv,
                    "Trapezoid", world_lv, false, 0, true);
    auto airbox0_sv = new G4Box("AirBox0", hole_y/2, trapezoid_height/2, hole_x/2);
    airbox0_lv = new G4LogicalVolume(airbox0_sv, fAir, "AirBox0");
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, 0), airbox0_lv, "AirBox0",
                    trapezoid_lv, false, 0); 
    
    // 定义90mm长束流口
    auto box1_sv = new G4Box("Box1", box1_x/2, box1_y/2, box1_z/2);
    box1_lv = new G4LogicalVolume(box1_sv, fPb, "Box1");
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, 46.0*mm + box1_z/2), box1_lv, "Box1",
                    world_lv, false, 0);
    auto airbox1_sv = new G4Box("AirBox1", hole_x/2, hole_y/2, box1_z/2);
    airbox1_lv = new G4LogicalVolume(airbox1_sv, fAir, "AirBox1");
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, 0), airbox1_lv, "AirBox1",
                    box1_lv, false, 0);    
    
    // 定义梯形台后方矩形台
    auto box2_sv = new G4Box("Box2", box2_x/2, box2_y/2, box2_z/2);
    box2_lv = new G4LogicalVolume(box2_sv, fPb, "Box2");
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, 30*mm + box2_z/2), box2_lv, "Box2",
                    world_lv, false, 0);  
    auto airbox2_sv = new G4Box("AirBox2", hole_x/2, hole_y/2, box2_z/2);
    airbox2_lv = new G4LogicalVolume(airbox2_sv, fAir, "AirBox2");
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, 0), airbox2_lv, "AirBox2",
                    box2_lv, false, 0);   

    // 定义射线源筒
    // 刀切面
    auto cyl = new G4Tubs("cyl", 0, 50*mm, 35*mm, 0*deg, 360*deg);
    auto cutBox1 = new G4Box("CutBox1", 100*mm, 100*mm, 100*mm);
    G4Transform3D tr1(G4RotationMatrix(), G4ThreeVector(36*mm + 100*mm, 0, 0)); 
    auto Box1Cut_sv = new G4SubtractionSolid("CylCut", cyl, cutBox1, tr1);  

    // 内部矩形空腔
    auto cutBox2 = new G4Box("CutBox2", box2_x/2, box2_y/2, box2_z/2 + 0.1*mm); 
    G4ThreeVector cutPos2(30*mm + box2_z/2 + 0.1*mm, 0, 0);
    rot = G4RotationMatrix();
    rot.rotateY(90*deg);
    G4Transform3D tr3(rot, cutPos2); 
    auto Box2Cut_sv = new G4SubtractionSolid("Box2Cut", Box1Cut_sv, cutBox2, tr3);
    cylCut_lv = new G4LogicalVolume(Box2Cut_sv, fPb, "CylCut");

    // // test
    // auto cyl = new G4Tubs("cyl", 0, 50*mm, 35*mm, 0*deg, 360*deg);
    // cylCut_lv = new G4LogicalVolume(cyl, fAir, "CylCut");

    // 内部锥形空腔
    auto trd = new G4Trd("trd", 0.1*mm, 8.0*mm, 0.1*mm, 34*mm, 16*mm);
    rot = G4RotationMatrix();
    rot.rotateY(90*deg);
    G4Transform3D tr2(rot, G4ThreeVector(14*mm, 0, 0));
    trd_lv = new G4LogicalVolume(trd, fAir, "trd");
    new G4PVPlacement(tr2, trd_lv,
                    "trd", cylCut_lv, false, 0, true);

    rot = G4RotationMatrix();
    rot.rotateY(-90*deg);
    new G4PVPlacement(G4Transform3D(rot, G4ThreeVector(0, 0, 0)), cylCut_lv,
                    "CylCut", world_lv, false, 0, true);

    // 放置铜环
    auto ring_sv = new G4Tubs("ring", innerRadius*mm, outerRadius*mm, halfLengthZ, 0.*deg, 360.*deg);
    ring_lv = new G4LogicalVolume(ring_sv, fCu, "ring");
    new G4PVPlacement(G4Transform3D(rot, G4ThreeVector(0, 0, 0)), ring_lv,
                    "ring", world_lv, false, 0, true);

    
    VisAttributes();
    
    return world_pv;
}

void DetectorConstruction::ConstructSDandField()
{
    G4SDManager* sdManager = G4SDManager::GetSDMpointer();
    auto sd = new AirSD("AirSD");
    sdManager->AddNewDetector(sd);
    sd_lv->SetSensitiveDetector(sd);
}

void DetectorConstruction::VisAttributes()
{
    auto world_va = new G4VisAttributes(G4Color(1.0, 1.0, 1.0, 1.0 / 255.0));
    world_lv->SetVisAttributes(world_va);
    sd_lv->SetVisAttributes(world_va);

    auto air_va = new G4VisAttributes(G4Color(1.0, 1.0, 1.0));
    airbox0_lv->SetVisAttributes(air_va);
    airbox1_lv->SetVisAttributes(air_va);
    airbox2_lv->SetVisAttributes(air_va);
    trd_lv->SetVisAttributes(air_va);

    auto box_va = new G4VisAttributes(G4Color(1.0, 0, 0, 200.0 / 255.0));
    box1_lv->SetVisAttributes(box_va);
    box2_lv->SetVisAttributes(box_va);
    trapezoid_lv->SetVisAttributes(box_va);

    auto cyl_va = new G4VisAttributes(G4Color(0.0, 170.0 / 255.0, 1.0, 100.0 / 255.0));
    cylCut_lv->SetVisAttributes(cyl_va);

    auto Cu_va = new G4VisAttributes(G4Color(184./255., 115./255., 51./255.));
    ring_lv->SetVisAttributes(Cu_va);
}