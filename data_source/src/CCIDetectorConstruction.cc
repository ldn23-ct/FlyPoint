#include "CCIDetectorConstruction.hh"
#include "CCIConfig.hh"
#include "CCILayerSD.hh"

#include "G4Material.hh"
#include "G4NistManager.hh"
#include "CADMesh.hh"
#include "G4Box.hh"
#include "G4Orb.hh"
#include "G4Tubs.hh"
#include "G4Sphere.hh"
#include "G4Trap.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4PVReplica.hh"
#include "G4GlobalMagFieldMessenger.hh"
#include "G4AutoDelete.hh"

#include "G4GeometryManager.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4SolidStore.hh"
#include "G4SubtractionSolid.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include "G4SDManager.hh"
#include "G4MultiFunctionalDetector.hh"
#include "G4VPrimitiveScorer.hh"
#include "G4PSEnergyDeposit.hh"

#include <fstream>


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4ThreadLocal
    G4GlobalMagFieldMessenger *CCIDetectorConstruction::fMagFieldMessenger = 0;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

CCIDetectorConstruction::CCIDetectorConstruction()
    : G4VUserDetectorConstruction(),
      fAbsorberPV(0),
      fScatterPV(0),
      fCheckOverlaps(true)
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

CCIDetectorConstruction::~CCIDetectorConstruction()
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume *CCIDetectorConstruction::Construct()
{
  // Define materials
  DefineMaterials();

  // Define volumes
  return DefineVolumes();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void CCIDetectorConstruction::DefineMaterials()
{
  G4NistManager *man = G4NistManager::Instance();
    G4bool isotopes = false;

    G4Element *Fe = man->FindOrBuildElement("Fe", isotopes);
    G4Element *Al = man->FindOrBuildElement("Al", isotopes);
    G4Element *Cd = man->FindOrBuildElement("Cd", isotopes);
    G4Element *Zn = man->FindOrBuildElement("Zn", isotopes);
    G4Element *Te = man->FindOrBuildElement("Te", isotopes);
    G4Element *Pb = man->FindOrBuildElement("Pb", isotopes);
    G4Element *H = man->FindOrBuildElement("H", isotopes);
    G4Element *O = man->FindOrBuildElement("O", isotopes);
    G4Element *N = man->FindOrBuildElement("N", isotopes);
    G4Element *C = man->FindOrBuildElement("C", isotopes);
    G4Element *Na = man->FindOrBuildElement("Na", isotopes);
    //G4Element *Mg = man->FindOrBuildElement("Mg", isotopes);
    G4Element *P = man->FindOrBuildElement("P", isotopes);
    G4Element *S = man->FindOrBuildElement("S", isotopes);
    G4Element *Cl = man->FindOrBuildElement("Cl", isotopes);
    G4Element *K = man->FindOrBuildElement("K", isotopes);
    G4Element *Ca = man->FindOrBuildElement("Ca", isotopes);
    G4Element *Sr = man->FindOrBuildElement("Sr", isotopes);
    G4Element *Zr = man->FindOrBuildElement("Zr", isotopes);
    G4Element *Si = man->FindOrBuildElement("Si", isotopes);
    G4Element *Ge = man->FindOrBuildElement("Ge", isotopes);
    G4Element *Lu = man->FindOrBuildElement("Lu", isotopes);
    G4Element *Y = man->FindOrBuildElement("Y", isotopes);
    G4Element *Ce = man->FindOrBuildElement("Ce", isotopes);
    G4Element *Bi = man->FindOrBuildElement("Bi", isotopes);
    G4Element *Gd = man->FindOrBuildElement("Gd", isotopes);
    G4Element *W = man->FindOrBuildElement("W", isotopes);
    G4Element *Mn = man->FindOrBuildElement("Mn", isotopes);
    G4Element *Cu = man->FindOrBuildElement("Cu", isotopes);
    G4Element *Cr = man->FindOrBuildElement("Cr", isotopes);
    G4Element *Ni = man->FindOrBuildElement("Ni", isotopes);
    G4Element *Ga = man->FindOrBuildElement("Ga", isotopes);

    // PMMA C5H8O2
    G4Material *PMMA = new G4Material("PMMA", 1.19 * g / cm3, 3);
    PMMA->AddElement(H, 8);
    PMMA->AddElement(C, 5);
    PMMA->AddElement(O, 2);

    // GAGG Gd3Al2Ga3O12
    G4Material *GAGG = new G4Material("GAGG", 6.60 * g / cm3, 4);
    GAGG->AddElement(Gd, 3);
    GAGG->AddElement(Al, 2);
    GAGG->AddElement(Ga, 3);
    GAGG->AddElement(O, 12);

    // C
    G4Material *Carbon = new G4Material("Carbon", 1.8 * g / cm3,1);
    Carbon->AddElement(C, 1);

    // Al
    man->FindOrBuildMaterial("G4_Al");

    // W
    man->FindOrBuildMaterial("G4_W");

    //Pb
    man->FindOrBuildMaterial("G4_Pb");

    // 三元乙丙 C8H6CL3NO
    G4Material *Acetamide = new G4Material("Acetamide", 1.22 * g / cm3, 5);
    Acetamide->AddElement(C, 8);
    Acetamide->AddElement(H, 6);
    Acetamide->AddElement(Cl, 3);
    Acetamide->AddElement(N, 1);
    Acetamide->AddElement(O, 1);

    // 聚氨酯 C3H8N2O
    G4Material *Polyurethane = new G4Material("PU", 0.9 * g / cm3, 4);
    Polyurethane->AddElement(C, 3);
    Polyurethane->AddElement(H, 8);
    Polyurethane->AddElement(N, 2);
    Polyurethane->AddElement(O, 1);

    // 硝酸铵 NH4NO3
    G4Material *AN = new G4Material("AN", 1.76 * g / cm3, 3);
    AN->AddElement(H, 4);
    AN->AddElement(N, 2);
    AN->AddElement(O, 3);

    // Air
    man->FindOrBuildMaterial("G4_AIR");


    // Print materials
    G4cout << *(G4Material::GetMaterialTable()) << G4endl;
}

G4VPhysicalVolume *CCIDetectorConstruction::DefineVolumes()
{
  // Get config instance
  auto config = CCIConfig::GetInstance();

  // Geometry parameters
    G4double worldSizeXY = 3000.*mm;
    G4double worldSizeZ = 3000.*mm;

    G4double object_x = config->GetObjectX();
    G4double object_y = config->GetObjectY();
    G4double object_z = config->GetObjectZ();

    G4int det_nx = config->GetDetNX();
    G4int det_ny = config->GetDetNY();
    G4double det_size = config->GetDetSize();
    G4double det_thick = config->GetDetThick();
    G4double det_pitch = config->GetDetPitch();
    
    G4ThreeVector det_center = config->GetDetCenter();
    G4double det_theta = config->GetDetTheta();


    G4double slit2det_dist = config->GetSlit2DetDist();
    G4double slit_thick = config->GetSlitThick();
    G4double slit_width = config->GetSlitWidth();
    G4double slit_height = config->GetSlitHeight();
    G4double inner_x = det_nx * det_pitch;
    G4double inner_y = det_ny * det_pitch;



    // Get materials
    auto defaultMaterial = G4Material::GetMaterial("G4_AIR");
    auto objectMaterial = G4Material::GetMaterial("G4_Al");
    auto objectMaterial_pb = G4Material::GetMaterial("G4_Pb");
    auto layerMaterial = G4Material::GetMaterial("GAGG");
    auto collimatorsMaterial = G4Material::GetMaterial("G4_W");
    auto leadMaterial = G4Material::GetMaterial("G4_W");


    //
    // World
    //
    auto worldS = new G4Box("World",                                           // its name
                            worldSizeXY / 2, worldSizeXY / 2, worldSizeZ / 2); // its size

    auto worldLV = new G4LogicalVolume(
        worldS,          // its solid
        defaultMaterial, // its material
        "World");        // its name

    auto worldPV = new G4PVPlacement(nullptr,         // no rotation
                                     G4ThreeVector(), // at (0,0,0)
                                     worldLV,         // its logical volume
                                     "World",         // its name
                                     nullptr,         // its mother  volume
                                     false,           // no boolean operation
                                     0,               // copy number
                                     fCheckOverlaps); // checking overlaps

    // 物体本体
    // auto objectS_solid = new G4Box("object_solid", object_x/2, object_y/2, object_z/2);

    // 在物体中心创建一个 10mm 边长的立方体空气空隙
    // G4double gap_size = 10.0 * mm;
    // auto gapS = new G4Box("gap", gap_size/2, gap_size/2, gap_size/2);

    // 这里空隙在原点，所以不需要旋转和平移
    // auto objectS = new G4SubtractionSolid("object_with_gap", objectS_solid, gapS, nullptr, G4ThreeVector());
    // auto objectS = objectS_solid;

    // new G4PVPlacement(nullptr, G4ThreeVector(0,0,0), objectLV, "object", worldLV, false, 0, fCheckOverlaps);


    // // 创建物体包络体和分层结构
    // // 1. 创建一个“母体”或“包络”实体，它将容纳所有的层。
    // auto objectEnvelopeS = new G4Box("objectEnvelopeS", object_x/2, object_y/2, object_z/2);
    // auto objectEnvelopeLV = new G4LogicalVolume(objectEnvelopeS, defaultMaterial, "objectEnvelopeLV");
    // objectEnvelopeLV->SetVisAttributes(G4VisAttributes::GetInvisible()); // 让包络体透明
    // new G4PVPlacement(nullptr, G4ThreeVector(0,0,0), objectEnvelopeLV, "Envelope", worldLV, false, 0, fCheckOverlaps);

    // // 2. 定义您想要使用的参数
    // G4double slice_thickness = 3.0 * mm;

    // // 3. 创建一个标准的实体物质层的逻辑体，用于所有完整的层
    // auto materialLayerS = new G4Box("materialLayerS", object_x/2, object_y/2, slice_thickness/2);
    // auto materialLayerLV = new G4LogicalVolume(materialLayerS, objectMaterial, "materialLayerLV");
    // auto materialVisAtt = new G4VisAttributes(G4Colour(0.5, 0.5, 0.5, 0.8)); // 设置为灰色实体
    // materialVisAtt->SetForceSolid(true);
    // materialLayerLV->SetVisAttributes(materialVisAtt);

    // // 4. 计算需要放置的实体层数量和位置
    // G4int total_layers = static_cast<G4int>(std::floor(object_z / slice_thickness));
    // G4int num_material_layers = static_cast<G4int>(std::ceil(total_layers / 2.0));
    // G4double end_z = object_z / 2.0;
    // G4double start_z = -object_z / 2.0;

    // for (int i = 0; i < num_material_layers; ++i) {
    //     // 计算当前实体层在未被截断时的中心位置
    //     G4double z_pos_ideal = end_z - (0.5 + i * 2.0) * slice_thickness;
    //     // 计算这一层理想的下边界
    //     G4double bottom_edge_ideal = z_pos_ideal - slice_thickness / 2.0;

    //     if (bottom_edge_ideal >= start_z) {
    //         // --- 情况A：这是一个完整的层，没有超出边界 ---
    //         new G4PVPlacement(nullptr,
    //                           G4ThreeVector(0, 0, z_pos_ideal),
    //                           materialLayerLV, // 使用标准尺寸的逻辑体
    //                           "materialobject",
    //                           objectEnvelopeLV,
    //                           false,
    //                           i);
    //     } else {
    //         // --- 情况B：这是最后一个层，并且它超出了边界，需要被截断 ---
    //         // 计算被截断后的实际厚度
    //         G4double top_edge_actual = z_pos_ideal + slice_thickness / 2.0;
    //         G4double new_thickness = top_edge_actual - start_z;

    //         if (new_thickness <= 1e-9 *mm) continue; // 如果厚度过小，则忽略

    //         // 计算被截断后的实际中心位置
    //         G4double z_pos_actual = start_z + new_thickness / 2.0;

    //         // 创建一个特殊尺寸的、被截断的层
    //         auto truncatedLayerS = new G4Box("truncatedLayerS", object_x/2, object_y/2, new_thickness/2);
    //         auto truncatedLayerLV = new G4LogicalVolume(truncatedLayerS, objectMaterial, "truncatedLayerLV");
    //         truncatedLayerLV->SetVisAttributes(materialVisAtt);

    //         new G4PVPlacement(nullptr,
    //                           G4ThreeVector(0, 0, z_pos_actual),
    //                           truncatedLayerLV, // 使用特殊尺寸的逻辑体
    //                           "truncatedMaterialobject",
    //                           objectEnvelopeLV,
    //                           false,
    //                           i);
    //     }
    // }

    // --- 物体定义：Al-Pb-Al 三层结构 ---
    // 1. 创建一个“母体”或“包络”实体，它将容纳所有的层。
    auto objectEnvelopeS = new G4Box("objectEnvelopeS", object_x/2, object_y/2, object_z/2);
    auto objectEnvelopeLV = new G4LogicalVolume(objectEnvelopeS, defaultMaterial, "objectEnvelopeLV");
    objectEnvelopeLV->SetVisAttributes(G4VisAttributes::GetInvisible()); // 让包络体透明
    new G4PVPlacement(nullptr, G4ThreeVector(0,0,0), objectEnvelopeLV, "Envelope", worldLV, false, 0, fCheckOverlaps);

    // 2. 定义两种材料的逻辑体模板
    auto alVisAtt = new G4VisAttributes(G4Colour(0.5, 0.5, 0.5, 0.5)); // 灰色
    alVisAtt->SetForceSolid(true);
    auto pbVisAtt = new G4VisAttributes(G4Colour(0.2, 0.2, 0.3, 0.8)); // 深蓝色
    pbVisAtt->SetForceSolid(true);

    // --- Helper function to place a layer ---
    auto placeLayer = [&](G4double thickness, G4double center_z, G4Material* mat, const G4String& name, int copyNo) {
        auto layerS = new G4Box(name+"_S", object_x/2, object_y/2, thickness/2);
        auto layerLV = new G4LogicalVolume(layerS, mat, name+"_LV");
        if (mat == objectMaterial) layerLV->SetVisAttributes(alVisAtt);
        else layerLV->SetVisAttributes(pbVisAtt);
        new G4PVPlacement(nullptr, G4ThreeVector(0, 0, center_z), layerLV, name, objectEnvelopeLV, false, copyNo);
    };

    // --- 在下面三种配置中选择一种，取消其注释即可 ---

    // --- 配置 1: Al(30mm) - Pb(10mm) - Al(30mm) ---
    placeLayer(30.*mm, 20.*mm,  objectMaterial,    "materialobject", 0); // Z: [5, 35]
    placeLayer(10.*mm, 0.*mm,   objectMaterial_pb, "material", 1); // Z: [-5, 5]
    placeLayer(30.*mm, -20.*mm, objectMaterial,    "materialobject", 2); // Z: [-35, -5]
    

    /*
    // --- 配置 2: Al(35mm) - Pb(10mm) - Al(25mm) ---
    placeLayer(35.*mm, 17.5*mm,  objectMaterial,    "materialobject", 0); // Z: [0, 35]
    placeLayer(10.*mm, -5.*mm,    objectMaterial_pb, "material", 1); // Z: [-10, 0]
    placeLayer(25.*mm, -22.5*mm, objectMaterial,    "materialobject", 2); // Z: [-35, -10]
    */

    /*
    // --- 配置 3: Al(32mm) - Pb(10mm) - Al(28mm) ---
    placeLayer(32.*mm, 19.*mm,   objectMaterial,    "materialobject", 0); // Z: [3, 35]
    placeLayer(10.*mm, -2.*mm,   objectMaterial_pb, "material", 1); // Z: [-7, 3]
    placeLayer(28.*mm, -21.*mm,  objectMaterial,    "materialobject", 2); // Z: [-35, -7]
    */


    // 探测器layer
    // 单元体积和逻辑体
    auto layerS = new G4Box("layer", det_thick/2, det_size/2, det_size/2); // x为厚度
    auto layerLV = new G4LogicalVolume(layerS, layerMaterial, "layer");

    // 阵列旋转
    G4RotationMatrix* detRot = new G4RotationMatrix();
    detRot->rotateY(det_theta); // 阵列法向与yz平面夹角

    for(int iy=0; iy<det_ny; ++iy) {
        for(int iz=0; iz<det_nx; ++iz) {
            double x = 0; // 厚度方向
            double y = -(iy - det_ny/2 + 0.5) * det_pitch;
            double z = (iz - det_nx/2 + 0.5) * det_pitch;
            G4ThreeVector localPos(x, y, z);
            G4ThreeVector globalPos = det_center + detRot->inverse() * localPos;
            new G4PVPlacement(
                detRot,
                globalPos,
                layerLV,
                "layer",
                worldLV,
                false,
                iy*det_nx+iz,
                fCheckOverlaps
            );
        }
    }

    // --- 板 1 ---
    // 根据您提供的顶点反向计算出的参数
    G4double lead1_half_x = 2.00 * mm;      // 厚度的一半 (4mm / 2)
    G4double lead1_half_y = 150.0 * mm;      // 高度的一半 (300mm / 2)
    G4double lead1_half_z = 55.0 * mm;     // 长度的一半
    G4ThreeVector lead1_pos(-72.07*mm, 0.0*mm, 81.01*mm);
    G4double lead1_angle = 50.6 * deg;     // 绕Y轴的旋转角度

    auto leadS1 = new G4Box("lead1_S", lead1_half_x, lead1_half_y, lead1_half_z);
    auto leadLV1 = new G4LogicalVolume(leadS1, leadMaterial, "leadLV1");
    G4RotationMatrix* rot1 = new G4RotationMatrix();
    rot1->rotateY(lead1_angle);
    new G4PVPlacement(rot1, lead1_pos, leadLV1, "lead1", worldLV, false, 0, fCheckOverlaps);


    // --- 板 2 ---
    // 根据您提供的顶点反向计算出的参数
    G4double lead2_half_x = 2.00 * mm;      // 厚度的一半 (4mm / 2)
    G4double lead2_half_y = 150.0 * mm;      // 高度的一半 (300mm / 2)
    G4double lead2_half_z = 50.0 * mm;     // 长度的一半
    G4ThreeVector lead2_pos(-41.74*mm, 0.0*mm, 96.43*mm);
    G4double lead2_angle = 18.64 * deg;    // 绕Y轴的旋转角度

    auto leadS2 = new G4Box("lead2_S", lead2_half_x, lead2_half_y, lead2_half_z);
    auto leadLV2 = new G4LogicalVolume(leadS2, leadMaterial, "leadLV2");
    G4RotationMatrix* rot2 = new G4RotationMatrix();
    rot2->rotateY(lead2_angle);
    new G4PVPlacement(rot2, lead2_pos, leadLV2, "lead2", worldLV, false, 1, fCheckOverlaps);


    //
    // Visualization attributes
    //
    worldLV->SetVisAttributes(G4VisAttributes::GetInvisible());

    auto simpleBoxVisAtt= new G4VisAttributes(G4Colour(1.0,0.0,0.0));
    simpleBoxVisAtt->SetVisibility(true);
    layerLV->SetVisAttributes(simpleBoxVisAtt);
    //
    // Always return the physical World
    //
    return worldPV;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void CCIDetectorConstruction::ConstructSDandField()
{
    // 保留/创建全局场
    G4ThreeVector fieldValue = G4ThreeVector();
    fMagFieldMessenger = new G4GlobalMagFieldMessenger(fieldValue);
    fMagFieldMessenger->SetVerboseLevel(1);
    G4AutoDelete::Register(fMagFieldMessenger);

    auto sdMan = G4SDManager::GetSDMpointer();

    // 使用自定义敏感探测器 LayerSD，若已存在则复用（适应 MT）
    G4VSensitiveDetector* foundSD = sdMan->FindSensitiveDetector("LayerSD");
    if (!foundSD) {
        // 需要包含 CCILayerSD.hh
        auto layerSD = new CCILayerSD("LayerSD");
        sdMan->AddNewDetector(layerSD);
        G4cout << "Created SD: " << layerSD->GetName() << G4endl;
        foundSD = layerSD;
    } else {
        G4cout << "Reusing existing SD: " << foundSD->GetName() << G4endl;
    }

    // Attach 到所有名字为 "layer" 的逻辑体
    auto lvStore = G4LogicalVolumeStore::GetInstance();
    G4int nAttached = 0;
    for (auto lv : *lvStore) {
        if (!lv) continue;
        if (lv->GetName() == "layer") {
            lv->SetSensitiveDetector(foundSD);
            ++nAttached;
            G4cout << "Attach SD to LV: " << lv << " name=" << lv->GetName() << G4endl;
        }
    }
    G4cout << "Total layer LVs attached: " << nAttached << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......


