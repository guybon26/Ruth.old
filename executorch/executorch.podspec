Pod::Spec.new do |s|
  s.name             = 'executorch'
  s.version          = '0.1.0'
  s.summary          = 'Mock ExecuTorch'
  s.homepage         = 'https://github.com/pytorch/executorch'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'PyTorch' => 'pytorch-dev@fb.com' }
  s.source           = { :git => 'https://github.com/pytorch/executorch.git', :tag => s.version.to_s }
  s.ios.deployment_target = '15.0'
  s.source_files = 'runtime/**/*.{h,cpp}'
  
  s.subspec 'backends' do |b|
    b.subspec 'coreml' do |c|
        c.source_files = 'backends/coreml/**/*.{h,cpp,mm}'
    end
    b.subspec 'mps' do |m|
        m.source_files = 'backends/mps/**/*.{h,cpp,mm}'
    end
  end
end
